#!/usr/bin/env python3
# Asks six AI models to point to things in the jaguar photo, as pixel coordinates, and scores each point
# against hand-made reference boxes. Run: python3 locate_test.py --rounds 10   (trial: --rounds 1)
import argparse
import base64
import getpass
import http.client
import itertools
import json
import math
import os
import random
import re
import ssl
import statistics
import sys
import time
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent.parent / "test_images" / "IMG_1.jpg"
CONFIG_FILE = HERE / "patriot_api_config.json"       # endpoint + model API names (never the key)
WIDTH, HEIGHT = 1024, 685
ROUNDS = 10
NO_IMAGE_ROUNDS = 1
PAUSE = 1.0
TIMEOUT = 180

# The answer list: boxes drawn by hand, (left, top, right, bottom) in pixels from the top-left corner.
BOXES = {
    "A": ("jaguar on the left", (0, 235, 495, 545)),
    "B": ("jaguar on the right (head and chest)", (540, 210, 810, 505)),
    "C": ("branch above the jaguars", (0, 0, 1024, 190)),
    "D": ("eye of the jaguar on the right", (722, 288, 754, 316)),
    "E": ("nose of the jaguar on the left", (380, 420, 450, 452)),
    "F": ("eye 1 of the jaguar on the left", (342, 344, 380, 374)),
    "G": ("eye 2 of the jaguar on the left", (440, 372, 474, 400)),
}

INSTRUCTION = (f"The image is {WIDTH} pixels wide and {HEIGHT} pixels tall; (0, 0) is the top-left corner. "
               "Answer only with the pixel coordinates of the centre, as (x, y). If there are several, give one "
               "(x, y) for each. If it is not in the image, answer: none.")

# (id, category, question, boxes that must each get one point (empty = correct answer is "none"), source)
QUESTIONS = [
    ("L1", "Location words", "Point to the jaguar on the left.", ["A"], "RefCOCO"),
    ("L2", "Location words", "Point to the jaguar on the right.", ["B"], "RefCOCO"),
    ("L3", "Relationship", "Point to the branch above the jaguars.", ["C"], "RefCOCO / RefCOCOg"),
    ("L4", "Object part", "Point to the eye of the jaguar on the right.", ["D"], "Point-It-Out"),
    ("L5", "Object part", "Point to the nose of the jaguar on the left.", ["E"], "Point-It-Out"),
    ("L6", "Several targets", "Point to both jaguars.", ["A", "B"], "gRefCOCO"),
    ("L7", "Counting by pointing", "Point to every visible eye.", ["D", "F", "G"], "Molmo (PixMo-Points)"),
    ("L8", "Counting by pointing", "Point to every visible paw.", [], "Molmo (PixMo-Points), POPE"),
    ("L9", "No target", "Point to the leopard.", [], "gRefCOCO"),
    ("L10", "No target", "Point to the jaguar's collar.", [], "gRefCOCO, POPE"),
]

MODELS = [   # (name, company, possible API names: the first that answers is used)
    ("Llama-4 Maverick", "Meta", ["Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-4-Maverick",
                                  "llama-4-maverick-17b-128e-instruct-fp8", "llama-4-maverick"]),
    ("GPT-5.4", "GPT", ["gpt-5.4", "gpt-5.4-chat", "GPT-5.4"]),
    ("GPT-4.1", "GPT", ["gpt-4.1", "GPT-4.1"]),
    ("GPT-4o", "GPT", ["gpt-4o", "GPT-4o"]),
    ("Claude Sonnet 4.6", "Claude", ["claude-sonnet-4-6", "claude-sonnet-4.6", "Claude-Sonnet-4.6"]),
    ("Claude Sonnet 4.5", "Claude", ["claude-sonnet-4-5", "claude-sonnet-4.5", "Claude-Sonnet-4.5",
                                     "claude-sonnet-4-5-20250929"]),
]
STYLES_OPENAI = ["v1", "v1-preview", "deploy-2025", "deploy-2024"]
STYLES_CLAUDE = ["anthropic"] + STYLES_OPENAI


# Setup
def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def get_endpoint(cfg):
    if not cfg.get("endpoint"):
        ep = input("Endpoint URL (from the Patriot AI API Projects page): ").strip().rstrip("/")
        if not ep.startswith("http"):
            sys.exit("That doesn't look like a URL (it should start with https://).")
        cfg["endpoint"] = ep
        save_config(cfg)
    return cfg["endpoint"]


def get_key():
    key = os.environ.get("PATRIOT_KEY", "").strip() or \
        getpass.getpass("Primary key (hidden: paste it and press Enter): ").strip()
    return key or sys.exit("No key entered.")


# One request
def build_request(style, endpoint, key, api_name, prompt, image, usage=True):
    first = f"Request id {uuid.uuid4().hex[:12]}."              # unique start: no cached shortcut
    if style == "anthropic":
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64",
                                                           "media_type": "image/jpeg", "data": image}})
        payload = {"model": api_name, "max_tokens": 1024, "stream": True, "system": first,
                   "messages": [{"role": "user", "content": content}]}
        return (f"{endpoint}/anthropic/v1/messages", payload,
                {"api-key": key, "x-api-key": key, "anthropic-version": "2023-06-01"})
    content = prompt if not image else [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
    payload = {"messages": [{"role": "system", "content": first}, {"role": "user", "content": content}],
               "stream": True}
    if usage:
        payload["stream_options"] = {"include_usage": True}
    dep = urllib.parse.quote(api_name, safe="")
    urls = {"v1": f"{endpoint}/openai/v1/chat/completions",
            "v1-preview": f"{endpoint}/openai/v1/chat/completions?api-version=preview",
            "deploy-2025": f"{endpoint}/openai/deployments/{dep}/chat/completions?api-version=2025-04-01-preview",
            "deploy-2024": f"{endpoint}/openai/deployments/{dep}/chat/completions?api-version=2024-10-21"}
    if style.startswith("v1"):
        payload["model"] = api_name
    return urls[style], payload, {"api-key": key}


def short_error(body):
    try:
        err = json.loads(body)
        err = err.get("error", err)
        return f"{err.get('code', '') or err.get('type', '')} {err.get('message', '')}".strip()[:300]
    except Exception:
        return (body or "").strip()[:300].replace("\n", " ")


def read_event(style, ev, r):
    """Returns the answer text in one streamed event and stores token counts in r."""
    if style == "anthropic":
        if ev.get("type") == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            r["cached"] = u.get("cache_read_input_tokens") or 0
            r["input_tokens"] = (u.get("input_tokens") or 0) + r["cached"]
        if ev.get("type") == "content_block_delta":
            return (ev.get("delta") or {}).get("text") or ""
        return ""
    u = ev.get("usage")
    if isinstance(u, dict):
        r["input_tokens"] = u.get("prompt_tokens")
        r["cached"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    choices = ev.get("choices") or []
    return ((choices[0].get("delta") or {}).get("content") or "") if choices else ""


def ask(style, endpoint, key, api_name, prompt, image=None, usage=True):
    """One request. The connection is opened first; the clock covers sending -> last word."""
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image, usage)
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    r = dict(ok=False, status=None, error="", text="", first=None, total=None)
    conn = None
    try:
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=TIMEOUT,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=TIMEOUT)
        conn.connect()                                               # DNS + TCP + TLS, not timed
        conn.putrequest("POST", u.path + (f"?{u.query}" if u.query else ""), skip_accept_encoding=True)
        for k, v in {**headers, "Content-Type": "application/json", "Accept": "text/event-stream",
                     "Content-Length": str(len(body))}.items():
            conn.putheader(k, v)
        t0 = time.perf_counter()                                     # clock starts
        conn.endheaders(body)
        resp = conn.getresponse()
        r["status"] = resp.status
        if resp.status >= 400:
            r["error"] = short_error(resp.read().decode("utf-8", "replace"))
            return r
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == "error" or (ev.get("error") and not ev.get("choices")):
                r["error"] = short_error(data)
                return r
            piece = read_event(style, ev, r)
            if piece:
                r["first"] = r["first"] or time.perf_counter() - t0   # first word
                r["text"] += piece
            if ev.get("type") == "message_stop":
                break
        r["total"] = time.perf_counter() - t0                        # last word
        r["ok"] = True
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    finally:
        if conn is not None:
            conn.close()


def call(model, endpoint, key, prompt, image=None):
    r = ask(model["style"], endpoint, key, model["api_name"], prompt, image, model.get("usage", True))
    if not r["ok"] and r["status"] == 400 and "stream_options" in r["error"].lower():
        model["usage"] = False                                       # this server rejects token counts
        r = ask(model["style"], endpoint, key, model["api_name"], prompt, image, False)
    return r


def find_models(endpoint, key, labels):
    print("Finding the models' API names...")
    found = {}
    for label, company, names in MODELS:
        if label not in labels:
            continue
        for name in names:
            for style in (STYLES_CLAUDE if company == "Claude" else STYLES_OPENAI):
                res = call({"api_name": name, "style": style}, endpoint, key, "Reply with: hi")
                if res["status"] == 401:
                    sys.exit("The key was rejected (401). Copy the primary key again.")
                if res["ok"]:
                    found[label] = {"api_name": name, "style": style}
                    break
            if label in found:
                break
        print(f"  {label:<18} {'-> ' + found[label]['api_name'] if label in found else 'NOT FOUND'}")
    return found


# Scoring (after the clock stops)
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
NUMBER_WORDS.update({w: str(i * 10) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety"
                                                          .split(), 2)})
CORRECT, WRONG_PLACE, WRONG_COUNT = "correct", "wrong place", "wrong number of points"
INVENTED, POINTED_ANYWAY = "pointed to something not there", "pointed anyway (said it isn't there)"
SAID_NONE, UNCLEAR, FAILED = "answered none", "unclear", "request failed"
MISTAKES = [WRONG_PLACE, WRONG_COUNT, INVENTED, POINTED_ANYWAY, SAID_NONE, UNCLEAR]


def read_points(text):
    """Every different (x, y) in the reply, in order. A box (x1, y1, x2, y2) counts as its centre.
    Number words are read as digits: one reply wrote (512, ninety)."""
    text = re.sub(r"[a-z]+", lambda m: NUMBER_WORDS.get(m.group(), m.group()), (text or "").lower())
    points = []
    for group in re.findall(r"[(\[]\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*[)\]]", text):
        n = [float(v) for v in group.split(",")]
        if len(n) == 2:
            points.append((n[0], n[1]))
        elif len(n) == 4:
            points.append(((n[0] + n[2]) / 2, (n[1] + n[3]) / 2))
    return list(dict.fromkeys(points))            # a point repeated at the end of a reply counts once


def inside(p, box):
    x1, y1, x2, y2 = BOXES[box][1]
    return x1 <= p[0] <= x2 and y1 <= p[1] <= y2


def centre(box):
    x1, y1, x2, y2 = BOXES[box][1]
    return (x1 + x2) / 2, (y1 + y2) / 2


def score(qid, text):
    """Returns (result, distance, points). Correct = one point inside each target box and no extra points,
    or "none" when nothing should be there. Distance = for each target, pixels from its box centre to the
    nearest point given, averaged."""
    question, targets = next((q[2], q[3]) for q in QUESTIONS if q[0] == qid)
    points = read_points(text)
    if not points:
        if not re.search(r"\bnone\b|\bnot\b|\bno\b", (text or "").lower()):
            return UNCLEAR, None, points
        return (CORRECT if not targets else SAID_NONE), None, points
    if not targets:
        thing = re.findall(r"[a-z]+", question.lower())[-1]          # leopard, paw, collar
        anyway = re.search(rf"\b(not|no) (an? )?{thing}", text.lower())
        return (POINTED_ANYWAY if anyway else INVENTED), None, points
    dist = statistics.mean(min(math.dist(centre(t), p) for p in points) for t in targets)
    if len(points) != len(targets):
        return WRONG_COUNT, dist, points
    if any(all(inside(p, t) for p, t in zip(order, targets)) for order in itertools.permutations(points)):
        return CORRECT, dist, points
    return WRONG_PLACE, dist, points


def expected(targets):
    if not targets:
        return "none"
    return " + ".join(f"1 point in box {t}" for t in targets)


def without_image_did(x):
    """What a model did when the question came without the photo."""
    if x["points"]:
        return "gave coordinates anyway"
    return "said none" if len(x["reply"].split()) <= 3 else "said it can't see an image"


# The test
def run(endpoint, key, models, image, rounds, no_image_rounds, pause):
    rows = []
    plan = [(False, r) for r in range(1, no_image_rounds + 1)] + [(True, r) for r in range(1, rounds + 1)]
    for with_image, rnd in plan:
        for qid, category, text, targets, source in random.sample(QUESTIONS, len(QUESTIONS)):
            for label in random.sample(list(models), len(models)):
                time.sleep(pause)
                r = call(models[label], endpoint, key, f"{text} {INSTRUCTION}", image if with_image else None)
                result, dist, points = score(qid, r["text"]) if r["ok"] else (FAILED, None, [])
                rows.append(dict(round=rnd, image=with_image, model=label, qid=qid, reply=r["text"] or r["error"],
                                 points=points, result=result, dist=dist, first=r["first"], total=r["total"],
                                 input_tokens=r.get("input_tokens"), cached=r.get("cached") or 0))
        print(f"{'Without image' if not with_image else 'Round'} {rnd}  ({datetime.now():%H:%M})")
        for label in models:
            mine = [x for x in rows if x["round"] == rnd and x["image"] == with_image and x["model"] == label]
            wrong = [x["qid"] for x in mine if x["result"] != CORRECT]
            print(f"  {label:<18} {len(mine) - len(wrong):>2}/{len(mine)} correct   {' '.join(wrong)}")
    return rows


def regrade(path):
    """Reads the replies saved in an earlier results file and scores them again with the rules above."""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    targets_of = {q[0]: q[3] for q in QUESTIONS}
    rows = []
    for r in wb["Answers"].iter_rows(values_only=True):
        if isinstance(r[0], int):
            result, dist, points = score(r[2], r[6])
            rows.append(dict(round=r[0], image=True, model=r[1], qid=r[2], reply=r[6] or "", points=points,
                             result=result, dist=dist, first=r[10], total=r[11]))
    if "Without image" in wb.sheetnames:
        for r in wb["Without image"].iter_rows(values_only=True):
            if r[1] in targets_of:
                result, dist, points = score(r[1], r[4])
                rows.append(dict(round=1, image=False, model=r[0], qid=r[1], reply=r[4] or "", points=points,
                                 result=result, dist=dist, first=None, total=None))
    return rows


# Excel
class Red:
    """Marks a cell to be filled red (an incorrect answer)."""
    def __init__(self, value):
        self.value = value


def add_table(ws, headers, rows, widths=None, title=None, formats=None, filters=False):
    from openpyxl.styles import Alignment, Font, PatternFill
    if title:
        ws.append([title])
        ws.cell(ws.max_row, 1).font = Font(bold=True, size=12)
    ws.append(headers)
    top = ws.max_row
    for c in ws[top]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="FFF2CC")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append([re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", v) if isinstance(v, str) else v
                   for v in (c.value if isinstance(c, Red) else c for c in row)])
        for i, c in enumerate(row, 1):
            cell = ws.cell(ws.max_row, i)
            if isinstance(c, Red):
                cell.fill = PatternFill("solid", fgColor="F4CCCC")
            if formats and i in formats and isinstance(cell.value, (int, float)):
                cell.number_format = formats[i]
    for i, w in enumerate(widths or [], 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    if filters:
        ws.auto_filter.ref = f"A{top}:{ws.cell(ws.max_row, len(headers)).coordinate}"
        ws.freeze_panes = f"A{top + 1}"
    ws.append([])


def result_path(name):
    path = HERE / f"{name}.xlsx"
    return path if not path.exists() else HERE / f"{name}_{datetime.now():%Y-%m-%d_%H%M}.xlsx"


def med(values, digits=2):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), digits) if values else None


def show(points):
    return ", ".join(f"({x:g}, {y:g})" for x, y in points) or "-"




def grid(ws, labels, rows, title, value, questions=QUESTIONS, fmt=None):
    """One row per question, one column per model; red where the model was wrong at least once."""
    lines = []
    for qid, _, text, _, _ in questions:
        cells = []
        for m in labels:
            these = [x for x in rows if x["model"] == m and x["qid"] == qid]
            v = value(these) if these else None
            cells.append(Red(v) if any(x["result"] != CORRECT for x in these) else v)
        lines.append([f"{qid}  {text}"] + cells)
    add_table(ws, ["Question"] + labels, lines, title=title,
              formats={i: fmt for i in range(2, 2 + len(labels))} if fmt else None)


def widths(ws, first=46, rest=14, n=7):
    ws.column_dimensions["A"].width = first
    for i in range(2, n + 1):
        ws.column_dimensions[chr(64 + i)].width = rest


def write_excel(path, rows, labels, image_name, rounds):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    main = [x for x in rows if x["image"] and x["result"] != FAILED]
    right = {m: sum(x["result"] == CORRECT for x in main if x["model"] == m) for m in labels}
    labels = sorted(labels, key=lambda m: -right[m])                  # most accurate model first

    ws = wb.active
    ws.title = "Summary"
    ws.append([f"AI models pointing to things in {image_name} ({WIDTH} x {HEIGHT} pixels) | {rounds} rounds | "
               f"red = wrong at least once"])
    ws.append(["The models can't mark images. They gave locations as pixel coordinates in text, which were "
               "compared with hand-made reference boxes (sheet Answer list)."])
    ws["A1"].font = Font(bold=True, size=12)
    ws.append([])
    add_table(ws, ["Model", "Correct", "Median distance from the target (px)", "Median first word (s)",
                   "Median total (s)"],
              [[m, right[m] / max(sum(x["model"] == m for x in main), 1),
                med([x["dist"] for x in main if x["model"] == m], 0),
                med([x["first"] for x in main if x["model"] == m]),
                med([x["total"] for x in main if x["model"] == m])] for m in labels],
              formats={2: "0%", 4: "0.00", 5: "0.00"})
    grid(ws, labels, main, f"Times correct, out of {rounds}; red = wrong at least once",
         lambda xs: sum(x["result"] == CORRECT for x in xs))
    add_table(ws, ["Model"] + MISTAKES,
              [[m] + [sum(x["result"] == k for x in main if x["model"] == m) for k in MISTAKES]
               for m in labels],
              title=f"Kinds of mistakes (number of answers, out of {rounds * len(QUESTIONS)} per model)")
    widths(ws)
    failed = len([x for x in rows if x["result"] == FAILED])
    if failed:
        ws.append([f"{failed} requests failed (network or server) and are left out."])

    ws = wb.create_sheet("Distance and time")
    grid(ws, labels, main, "Median distance from the target box centre (px); questions with a target only",
         lambda xs: med([x["dist"] for x in xs], 0), questions=[q for q in QUESTIONS if q[3]])
    grid(ws, labels, main, "Median total time (s): from sending the question to the last word",
         lambda xs: med([x["total"] for x in xs]), fmt="0.00")
    widths(ws)

    ws = wb.create_sheet("Answers")
    text_of = {q[0]: q for q in QUESTIONS}
    order = {m: i for i, m in enumerate(labels)}
    lines = []
    for x in sorted((x for x in rows if x["image"]),
                    key=lambda x: (x["round"], order.get(x["model"], 99), int(x["qid"][1:]))):
        q = text_of[x["qid"]]
        line = [x["round"], x["model"], x["qid"], q[2], expected(q[3]), x["reply"], show(x["points"]), x["result"],
                med([x["dist"]], 0), med([x["first"]], 3), med([x["total"]], 3)]
        lines.append([Red(v) for v in line] if x["result"] not in (CORRECT, FAILED) else line)
    add_table(ws, ["Round", "Model", "Q", "Question", "Expected answer", "Answer given", "Points read", "Result",
                   "Distance (px)", "First word (s)", "Total (s)"], lines,
              [7, 18, 5, 36, 18, 45, 24, 22, 10, 10, 10], formats={10: "0.000", 11: "0.000"}, filters=True)

    ws = wb.create_sheet("Questions")
    add_table(ws, ["Q", "Category", "Question", "Expected answer", "Idea taken from"],
              [[qid, c, t, expected(tg), s] for qid, c, t, tg, s in QUESTIONS], [5, 20, 45, 28, 28],
              title=f"Every question was followed by: {INSTRUCTION}")

    ws = wb.create_sheet("Answer list")
    add_table(ws, ["Box", "What it covers", "Left", "Top", "Right", "Bottom"],
              [[b, name, *xy] for b, (name, xy) in BOXES.items()], [6, 36, 8, 8, 8, 8],
              title="Reference boxes in pixels, measured from the top-left corner. A point counts if it is "
                    "inside the box (edges included).")

    blind = [x for x in rows if not x["image"] and x["result"] != FAILED]
    if blind:
        ws = wb.create_sheet("Without image")
        did = ["gave coordinates anyway", "said none", "said it can't see an image"]
        add_table(ws, ["Model"] + did + ["Coordinates that landed in the right box (a lucky guess)"],
                  [[m] + [sum(without_image_did(x) == d for x in blind if x["model"] == m) for d in did] +
                   [sum(x["result"] == CORRECT and x["points"] != [] for x in blind if x["model"] == m)]
                   for m in labels], [24, 14, 14, 14, 20],
                  title="Each question sent once WITHOUT the photo. The honest answer is 'none' or 'I can't see "
                        "an image'. Coordinates here were made up.")
        add_table(ws, ["Model", "Q", "Question", "Answer given", "What it did", "Points read", "Would count as"],
                  [[x["model"], x["qid"], text_of[x["qid"]][2], x["reply"], without_image_did(x), show(x["points"]),
                    x["result"]] for x in sorted(blind, key=lambda x: (order.get(x["model"], 99), int(x["qid"][1:])))],
                  [18, 5, 36, 50, 22, 22, 22], filters=True)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="AI models point to things in the jaguar photo")
    ap.add_argument("--models", default="all", help='"all" or e.g. "GPT-4o,Llama-4 Maverick"')
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--no-image-rounds", type=int, default=NO_IMAGE_ROUNDS)
    ap.add_argument("--pause", type=float, default=PAUSE, help="seconds between requests")
    ap.add_argument("--regrade", metavar="FILE", help="score the replies in an earlier results file again")
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Please install openpyxl:  python3 -m pip install openpyxl")

    if args.regrade:
        rows = regrade(args.regrade)
        labels = [m[0] for m in MODELS if any(x["model"] == m[0] for x in rows)]
        path = result_path("locate_results")
        write_excel(path, rows, labels, IMAGE.name, max(x["round"] for x in rows))
        print(f"Saved to {path}")
        return

    if not IMAGE.is_file():
        sys.exit(f"Image not found: {IMAGE}")
    image = base64.b64encode(IMAGE.read_bytes()).decode()
    names = [m[0] for m in MODELS]
    labels = names if args.models == "all" else [m.strip() for m in args.models.split(",")]
    if any(l not in names for l in labels):
        sys.exit(f"Unknown model. Use: {', '.join(names)}")
    cfg = load_config()
    endpoint, key = get_endpoint(cfg), get_key()
    found = cfg.get("found", {})
    missing = [l for l in labels if l not in found]
    if missing:
        found.update(find_models(endpoint, key, missing))
        cfg["found"] = found
        save_config(cfg)
    models = {l: dict(found[l]) for l in labels if l in found}
    if not models:
        sys.exit("No model answered.")

    n = (args.rounds + args.no_image_rounds) * len(QUESTIONS) * len(models)
    print(f"{len(models)} models x {len(QUESTIONS)} questions x {args.rounds} rounds "
          f"(+{args.no_image_rounds} without image) = {n} requests, about {n * (args.pause + 2) / 60:.0f} min")
    rows = run(endpoint, key, models, image, args.rounds, args.no_image_rounds, args.pause)

    path = result_path("locate_results")
    write_excel(path, rows, list(models), IMAGE.name, args.rounds)
    main_rows = [x for x in rows if x["image"] and x["result"] != FAILED]
    for m in models:
        mine = [x for x in main_rows if x["model"] == m]
        sizes = {}
        for x in mine:
            sizes.setdefault(x["qid"], []).append(x["input_tokens"] or 0)
        if any(x["cached"] for x in mine) or any(max(s) - min(s) > 20 for s in sizes.values()):
            print(f"WARNING {m}: cached work or changing input size; answers may not be independent.")
        print(f"{m:<18} {sum(x['result'] == CORRECT for x in mine) / max(len(mine), 1):.1%} correct, "
              f"median {med([x['total'] for x in mine])} s")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
