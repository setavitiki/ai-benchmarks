#!/usr/bin/env python3
# Asks six AI models 24 questions about one photo and times every answer.
# Run: python3 ai_models_test.py --rounds 25    (trial: --rounds 1 --no-image-rounds 0)
import argparse
import base64
import getpass
import http.client
import json
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
IMAGE = HERE.parent.parent / "test_images" / "IMG_1.jpg"      # Projects/test_images/IMG_1.jpg
CONFIG_FILE = HERE / "patriot_api_config.json"       # endpoint + model API names (never the key)
ROUNDS = 25
NO_IMAGE_ROUNDS = 1
PAUSE = 1.0                                          # seconds between requests
TIMEOUT = 180

# Questions
YN = "Answer with only yes or no."
ONE = "Answer with one word."
NUM = "Answer with a number."
AB = "Answer with one of the two options."
NUMBERS = {str(n) for n in range(21)}
COLORS = {"black", "white", "brown", "red", "blue", "green", "yellow", "orange", "gold", "golden",
          "grey", "gray", "silver", "pink", "purple", "tan", "beige", "leather"}


def q(id, category, text, instr, answer, accept, wrong):
    return dict(id=id, category=category, text=text, prompt=f"{text} {instr}", answer=answer,
                accept=set(accept), wrong=set(wrong))


# A reply is correct if its answer word is in the first set, incorrect if it is in the second.
QUESTIONS = [
    q("Q1", "Recognition", "What animal is in this image?", ONE, "jaguar", {"jaguar", "jaguars"},
      {"leopard", "leopards", "cheetah", "cheetahs", "tiger", "tigers", "lion", "lions", "puma",
       "cougar", "lynx", "ocelot"}),
    q("Q2", "Existence", "Is there a leopard in this image?", YN, "no", {"no", "not"}, {"yes"}),
    q("Q3", "Counting", "How many jaguars are in this image?", NUM, "2", {"2"}, NUMBERS - {"2"}),
    q("Q4", "Counting", "How many eyes are visible in this image?", NUM, "3", {"3"}, NUMBERS - {"3"}),
    q("Q5", "Counting", "How many paws are visible in this image?", NUM, "0", {"0", "none", "no"},
      NUMBERS - {"0"}),
    q("Q6", "Position", "What is above the jaguars?", ONE, "branch",
      {"branch", "branches", "tree", "trees", "log", "logs", "wood", "wooden", "trunk", "bark", "limb"},
      {"rock", "rocks", "stone", "stones", "sky", "nothing", "none", "ceiling"}),
    q("Q7", "Reasoning", "Are the jaguars awake?", YN, "yes", {"yes", "awake"}, {"no", "asleep", "sleeping"}),
    q("Q8", "Hallucination", "What color is the jaguar's collar?", ONE, "none (no collar)",
      {"none", "no", "not", "nothing", "neither"}, COLORS),
    q("Q9", "Hallucination", "Is the cat in this image white or black?", ONE, "neither (golden, spotted)",
      {"neither", "none", "golden", "gold", "yellow", "orange", "tan", "tawny", "brown", "spotted"},
      {"white", "black", "both"}),
    q("Q10", "Logic", "Is there a jaguar and a tiger in this image?", YN, "no", {"no", "not"}, {"yes"}),
    q("Q11", "Existence", "Is there grass in this image?", YN, "yes", {"yes"}, {"no"}),
    q("Q12", "Hallucination", "Is there a person in this image?", YN, "no", {"no", "not"}, {"yes"}),
    q("Q13", "Color", "Are the jaguars' spots black?", YN, "yes", {"yes"}, {"no"}),
    q("Q14", "Color", "Is the jaguars' fur mainly grey?", YN, "no", {"no", "not"}, {"yes"}),
    q("Q15", "Attribute", "Is the jaguars' fur striped or spotted?", AB, "spotted",
      {"spotted", "spots", "spot", "rosettes"}, {"striped", "stripes", "stripe"}),
    q("Q16", "Activity", "Are the jaguars lying down?", YN, "yes", {"yes", "lying"}, {"no"}),
    q("Q17", "Activity", "Are the jaguars running?", YN, "no", {"no", "not"}, {"yes"}),
    q("Q18", "Scene", "Are the jaguars on grass or on snow?", AB, "grass", {"grass", "grassy"}, {"snow"}),
    q("Q19", "Spatial relation", "Are the two jaguars lying next to each other?", YN, "yes", {"yes"}, {"no"}),
    q("Q20", "Comparison", "Are both animals the same species?", YN, "yes", {"yes"}, {"no"}),
    q("Q21", "Knowledge", "On which continent do these animals live in the wild?", ONE,
      "America (South, Central or North)", {"america", "americas", "south", "central", "north"},
      {"africa", "asia", "europe", "australia", "oceania", "antarctica"}),
    q("Q22", "Image style", "Is this image a photograph or a drawing?", AB, "photograph",
      {"photograph", "photo", "photographic"}, {"drawing", "painting", "illustration", "sketch"}),
    q("Q23", "Text (OCR)", "Is there any text or writing in this image?", YN, "no", {"no", "not"}, {"yes"}),
    q("Q24", "Logic", "Is there a jaguar and a tree branch in this image?", YN, "yes", {"yes"}, {"no"}),
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
    if not key:
        sys.exit("No key entered.")
    return key


def load_image(path):
    path = Path(path).expanduser()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(path.suffix.lower())
    if not path.is_file() or not mime:
        sys.exit(f"Image not found or not .jpg/.png: {path}")
    return path.name, (mime, base64.b64encode(path.read_bytes()).decode())


# One request
def build_request(style, endpoint, key, api_name, prompt, image, usage=True):
    first = f"Request id {uuid.uuid4().hex[:12]}."              # unique start: no cached shortcut
    if style == "anthropic":
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64",
                                                           "media_type": image[0], "data": image[1]}})
        payload = {"model": api_name, "max_tokens": 1024, "stream": True, "system": first,
                   "messages": [{"role": "user", "content": content}]}
        return (f"{endpoint}/anthropic/v1/messages", payload,
                {"api-key": key, "x-api-key": key, "anthropic-version": "2023-06-01"})
    content = prompt if not image else [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{image[0]};base64,{image[1]}"}}]
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
        t0 = time.perf_counter()                                     # ===== clock starts =====
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
                r["first"] = r["first"] or time.perf_counter() - t0   # first word arrived
                r["text"] += piece
            if ev.get("type") == "message_stop":
                break
        r["total"] = time.perf_counter() - t0                        # ===== clock stops: last word =====
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


# Grading (after the clock stops)
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def words(text):
    return [NUMBER_WORDS.get(w, w) for w in re.findall(r"[a-z]+|\d+", (text or "").lower())]


def grade(q, text):
    """'yes', 'no' or 'unclear'. Looks for an explicit answer ("Answer: 3"), then bold text,
    then the whole reply, and uses the first answer word it finds."""
    places = [m.group(1) for m in re.finditer(r"answer(?: would be| is)?\s*[:\-]?\s*([^\n.]{0,40})",
                                              text or "", re.I)]
    places += re.findall(r"\*\*([^*]+)\*\*", text or "") + [text or ""]
    for place in places:
        for w in words(place):
            if w in q["accept"]:
                return "yes"
            if w in q["wrong"]:
                return "no"
    return "unclear"


# The test
def run(endpoint, key, models, image, rounds, no_image_rounds, pause):
    rows = []
    plan = [(False, r) for r in range(1, no_image_rounds + 1)] + [(True, r) for r in range(1, rounds + 1)]
    for with_image, rnd in plan:
        for q in random.sample(QUESTIONS, len(QUESTIONS)):            # new order every round
            for label in random.sample(list(models), len(models)):   # every model, random order
                time.sleep(pause)
                r = call(models[label], endpoint, key, q["prompt"], image if with_image else None)
                correct = grade(q, r["text"]) if r["ok"] else "failed"
                if not with_image and r["ok"] and len(words(r["text"])) > 3:
                    correct = "no answer"                            # e.g. "I can't see an image"
                rows.append(dict(round=rnd, image=with_image, model=label, q=q, reply=r["text"] or r["error"],
                                 correct=correct, first=r["first"], total=r["total"],
                                 input_tokens=r.get("input_tokens"), cached=r.get("cached") or 0))
        print(f"{'Without image' if not with_image else 'Round'} {rnd}  ({datetime.now():%H:%M})")
        for label in models:
            mine = [x for x in rows if x["round"] == rnd and x["image"] == with_image and x["model"] == label]
            wrong = [x["q"]["id"] for x in mine if x["correct"] != "yes"]
            print(f"  {label:<18} {len(mine) - len(wrong):>2}/{len(mine)} correct   {' '.join(wrong)}")
    return rows


# Excel (shared with yolo_test.py)
class Red:
    """Marks a cell to be filled red (an incorrect answer)."""
    def __init__(self, value):
        self.value = value


def add_table(ws, headers, rows, widths=None, title=None, formats=None, filters=False):
    """Writes a table: optional bold title, a header row, the rows (Red cells filled red).
    formats = {column number: Excel number format}; filters = add filter buttons to the header."""
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


def write_excel(path, rows, labels, image_name, rounds):
    from openpyxl import Workbook
    wb = Workbook()
    main = [x for x in rows if x["image"] and x["correct"] != "failed"]
    score = {m: sum(x["correct"] == "yes" for x in main if x["model"] == m) for m in labels}
    labels = sorted(labels, key=lambda m: -score[m])                  # most accurate model first
    ws = wb.active
    ws.title = "Summary"
    table = []
    for m in labels:
        mine = [x for x in main if x["model"] == m]
        wrong = []
        for q in QUESTIONS:
            these = [x for x in mine if x["q"]["id"] == q["id"]]
            n = sum(x["correct"] != "yes" for x in these)
            if n:
                wrong.append(f"{q['id']} ({n} of {len(these)})")
        table.append([m, sum(x["correct"] == "yes" for x in mine) / max(len(mine), 1),
                      ", ".join(wrong) or "-", med([x["first"] for x in mine]), med([x["total"] for x in mine])])
    add_table(ws, ["Model", "Correct", "Wrong answers (times wrong)", "Median first word (s)",
                   "Median total (s)"], table, [30, 12, 55, 14, 14],
              title=f"AI models | image {image_name} | {rounds} rounds | times in seconds from sending the "
                    f"question to the first / last word | red = answered wrong at least once",
              formats={2: "0.0%", 4: "0.00", 5: "0.00"})
    grid = []
    for q in QUESTIONS:
        cells = []
        for m in labels:
            these = [x for x in main if x["model"] == m and x["q"]["id"] == q["id"]]
            t = med([x["total"] for x in these])
            cells.append(Red(t) if any(x["correct"] != "yes" for x in these) else t)
        grid.append([q["id"], q["category"], q["text"]] + cells)
    add_table(ws, ["Q", "Category", "Question"] + labels, grid,
              title="Median total time per question (s); red = this model got it wrong at least once",
              formats={i: "0.00" for i in range(4, 4 + len(labels))})
    failed = len([x for x in rows if x["correct"] == "failed"])
    if failed:
        ws.append([f"{failed} requests failed (network or server) and are left out."])

    ws = wb.create_sheet("Answers")
    verdict = {"yes": "yes", "no": "no", "unclear": "no (unclear answer)", "failed": "request failed"}
    lines = []
    order = {m: i for i, m in enumerate(labels)}
    for x in sorted((x for x in rows if x["image"]),
                    key=lambda x: (x["round"], order.get(x["model"], 99), int(x["q"]["id"][1:]))):
        line = [x["round"], x["model"], x["q"]["id"], x["q"]["category"], x["q"]["text"], x["q"]["answer"],
                x["reply"], verdict[x["correct"]], med([x["first"]], 3), med([x["total"]], 3)]
        lines.append([Red(v) for v in line] if x["correct"] in ("no", "unclear") else line)
    add_table(ws, ["Round", "Model", "Q", "Category", "Question", "Expected answer", "Answer given",
                   "Correct?", "First word (s)", "Total (s)"], lines, [7, 18, 5, 15, 45, 18, 45, 10, 10, 10],
              formats={9: "0.000", 10: "0.000"}, filters=True)

    ws = wb.create_sheet("Questions")
    add_table(ws, ["Q", "Category", "Question (as sent)", "Expected answer"],
              [[q["id"], q["category"], q["prompt"], q["answer"]] for q in QUESTIONS], [5, 16, 75, 30])

    blind = [x for x in rows if not x["image"]]
    if blind:
        ws = wb.create_sheet("Without image")
        result = {"yes": "right without the image", "no": "wrong", "unclear": "wrong",
                  "no answer": "no answer (said it can't see an image)", "failed": "request failed"}
        add_table(ws, ["Model", "Q", "Question", "Expected answer", "Answer given", "Result"],
                  [[x["model"], x["q"]["id"], x["q"]["text"], x["q"]["answer"], x["reply"], result[x["correct"]]]
                   for x in sorted(blind, key=lambda x: (int(x["q"]["id"][1:]), x["model"]))],
                  [18, 5, 45, 18, 45, 30],
                  title="Each question sent once WITHOUT the image. 'Right without the image' = answerable from "
                        "the question text alone, so a correct answer there says little about seeing.",
                  filters=True)
    wb.save(path)


# Main
def main():
    ap = argparse.ArgumentParser(description="1 image, 24 questions, several AI models")
    ap.add_argument("--models", default="all", help='"all" or e.g. "GPT-4o,Llama-4 Maverick"')
    ap.add_argument("--image", default=str(IMAGE))
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--no-image-rounds", type=int, default=NO_IMAGE_ROUNDS)
    ap.add_argument("--pause", type=float, default=PAUSE, help="seconds between requests")
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Please install openpyxl:  python3 -m pip install openpyxl")

    names = [m[0] for m in MODELS]
    labels = names if args.models == "all" else [m.strip() for m in args.models.split(",")]
    if any(l not in names for l in labels):
        sys.exit(f"Unknown model. Use: {', '.join(names)}")
    image_name, image = load_image(args.image)
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
    print(f"{image_name}: {len(models)} models x {len(QUESTIONS)} questions x {args.rounds} rounds "
          f"(+{args.no_image_rounds} without image) = {n} requests, about {n * (args.pause + 1.6) / 3600:.1f} h")
    rows = run(endpoint, key, models, image, args.rounds, args.no_image_rounds, args.pause)

    path = result_path("ai_models_results")
    write_excel(path, rows, list(models), image_name, args.rounds)
    main_rows = [x for x in rows if x["image"] and x["correct"] != "failed"]
    for m in models:
        mine = [x for x in main_rows if x["model"] == m]
        sizes = {}
        for x in mine:
            sizes.setdefault(x["q"]["id"], []).append(x["input_tokens"] or 0)
        if any(x["cached"] for x in mine) or any(max(s) - min(s) > 20 for s in sizes.values()):
            print(f"WARNING {m}: cached work or changing input size; answers may not be independent.")
        print(f"{m:<18} {sum(x['correct'] == 'yes' for x in mine) / max(len(mine), 1):.1%} correct, "
              f"median {med([x['total'] for x in mine])} s")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
