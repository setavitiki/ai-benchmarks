#!/usr/bin/env python3
# Experiment 06: 24 questions about one photo, all six models, 25 rounds.
# Run: python3 test_06.py --rounds 25    (trial: --rounds 1 --no-image-rounds 0)
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
from collections import Counter
from datetime import datetime
from pathlib import Path

# Settings
IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "test_images"
HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "patriot_api_config.json"       # endpoint + found API names (never the key)
IMAGE = "IMG_1"
ROUNDS = 25
NO_IMAGE_ROUNDS = 1
PAUSE = 1.0                                          # seconds between requests
TIMEOUT = 180

YN = "Answer with only yes or no."
ONE = "Answer with one word."
NUM = "Answer with a number."
AB = "Answer with one of the two options."
NUMBERS = {str(n) for n in range(21)}
COLORS = {"black", "white", "brown", "red", "blue", "green", "yellow", "orange", "gold", "golden",
          "grey", "gray", "silver", "pink", "purple", "tan", "beige", "leather"}


def q(id, text, instr, category, answer, accept, wrong):
    return dict(id=id, text=text, instr=instr, category=category, answer=answer,
                accept=set(accept), wrong=set(wrong))


# Accepted words count as correct, wrong words as wrong, anything else is marked "review".
QUESTIONS = [
    q("Q1", "What animal is in this image?", ONE, "Recognition", "jaguar", {"jaguar", "jaguars"},
      {"leopard", "leopards", "cheetah", "cheetahs", "tiger", "tigers", "lion", "lions", "puma",
       "cougar", "lynx", "ocelot"}),
    q("Q2", "Is there a leopard in this image?", YN, "Existence", "no", {"no", "not"}, {"yes"}),
    q("Q3", "How many jaguars are in this image?", NUM, "Counting", "2", {"2"}, NUMBERS - {"2"}),
    q("Q4", "How many eyes are visible in this image?", NUM, "Counting", "3", {"3"}, NUMBERS - {"3"}),
    q("Q5", "How many paws are visible in this image?", NUM, "Counting", "0", {"0", "none", "no"},
      NUMBERS - {"0"}),
    q("Q6", "What is above the jaguars?", ONE, "Position", "branch",
      {"branch", "branches", "tree", "trees", "log", "logs", "wood", "wooden", "trunk", "bark", "limb"},
      {"rock", "rocks", "stone", "stones", "sky", "nothing", "none", "ceiling"}),
    q("Q7", "Are the jaguars awake?", YN, "Reasoning", "yes", {"yes", "awake"}, {"no", "asleep", "sleeping"}),
    q("Q8", "What color is the jaguar's collar?", ONE, "Hallucination", "none (no collar)",
      {"none", "no", "not", "nothing", "neither"}, COLORS),
    q("Q9", "Is the cat in this image white or black?", ONE, "Hallucination", "neither (golden, spotted)",
      {"neither", "none", "golden", "gold", "yellow", "orange", "tan", "tawny", "brown", "spotted"},
      {"white", "black", "both"}),
    q("Q10", "Is there a jaguar and a tiger in this image?", YN, "Logic", "no", {"no", "not"}, {"yes"}),
    q("Q11", "Is there grass in this image?", YN, "Existence", "yes", {"yes"}, {"no"}),
    q("Q12", "Is there a person in this image?", YN, "Hallucination", "no", {"no", "not"}, {"yes"}),
    q("Q13", "Are the jaguars' spots black?", YN, "Color", "yes", {"yes"}, {"no"}),
    q("Q14", "Is the jaguars' fur mainly grey?", YN, "Color", "no", {"no", "not"}, {"yes"}),
    q("Q15", "Is the jaguars' fur striped or spotted?", AB, "Attribute", "spotted",
      {"spotted", "spots", "spot", "rosettes"}, {"striped", "stripes", "stripe"}),
    q("Q16", "Are the jaguars lying down?", YN, "Activity", "yes", {"yes", "lying"}, {"no"}),
    q("Q17", "Are the jaguars running?", YN, "Activity", "no", {"no", "not"}, {"yes"}),
    q("Q18", "Are the jaguars on grass or on snow?", AB, "Scene", "grass", {"grass", "grassy"}, {"snow"}),
    q("Q19", "Are the two jaguars lying next to each other?", YN, "Spatial relation", "yes", {"yes"}, {"no"}),
    q("Q20", "Are both animals the same species?", YN, "Comparison", "yes", {"yes"}, {"no"}),
    q("Q21", "On which continent do these animals live in the wild?", ONE, "Knowledge",
      "America (South, Central or North)", {"america", "americas", "south", "central", "north"},
      {"africa", "asia", "europe", "australia", "oceania", "antarctica"}),
    q("Q22", "Is this image a photograph or a drawing?", AB, "Image style", "photograph",
      {"photograph", "photo", "photographic"}, {"drawing", "painting", "illustration", "sketch"}),
    q("Q23", "Is there any text or writing in this image?", YN, "Text (OCR)", "no", {"no", "not"}, {"yes"}),
    q("Q24", "Is there a jaguar and a tree branch in this image?", YN, "Logic", "yes", {"yes"}, {"no"}),
]

MODELS = [   # (name, company, possible API names - the first that answers is used)
    ("Llama-4 Maverick", "Meta", ["Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-4-Maverick",
                                  "llama-4-maverick-17b-128e-instruct-fp8", "llama-4-maverick",
                                  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"]),
    ("GPT-5.4", "GPT", ["gpt-5.4", "gpt-5.4-chat", "GPT-5.4", "gpt-54"]),
    ("GPT-4.1", "GPT", ["gpt-4.1", "GPT-4.1", "gpt-41"]),
    ("GPT-4o", "GPT", ["gpt-4o", "GPT-4o"]),
    ("Claude Sonnet 4.6", "Claude", ["claude-sonnet-4-6", "claude-sonnet-4.6", "Claude-Sonnet-4.6",
                                     "claude-4-6-sonnet", "claude-sonnet-4-6-v1"]),
    ("Claude Sonnet 4.5", "Claude", ["claude-sonnet-4-5", "claude-sonnet-4.5", "Claude-Sonnet-4.5",
                                     "claude-sonnet-4-5-20250929", "claude-4-5-sonnet"]),
]

STYLES_OPENAI = ["v1", "v1-preview", "deploy-2025", "deploy-2024"]
STYLES_CLAUDE = ["anthropic"] + STYLES_OPENAI
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
NO_USAGE = set()            # (api_name, style) pairs whose server rejects stream_options


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
        ep = input("Endpoint URL (from the API Projects page): ").strip().rstrip("/")
        if not ep.startswith("http"):
            sys.exit("That doesn't look like a URL (it should start with https://). Run again.")
        cfg["endpoint"] = ep
        save_config(cfg)
    return cfg["endpoint"].rstrip("/")


def get_key():
    key = os.environ.get("PATRIOT_KEY", "").strip() or \
        getpass.getpass("Primary key (hidden: paste it, nothing will show, press Enter): ").strip()
    if not key:
        sys.exit("No key entered. Run again and paste the primary key.")
    return key


def load_image(name):
    path = Path(name).expanduser()
    if not path.is_file() and IMAGES_DIR.is_dir():
        path = next((f for f in sorted(IMAGES_DIR.iterdir())
                     if f.is_file() and f.stem.upper() == name.upper()), path)
    if not path.is_file():
        sys.exit(f"Image '{name}' not found (looked in {IMAGES_DIR}).")
    if path.suffix.lower() not in MIME:
        sys.exit(f"{path.name}: please save the image as .jpg or .png")
    return path.name, (MIME[path.suffix.lower()], base64.b64encode(path.read_bytes()).decode())


# One request
def build_request(style, endpoint, key, api_name, prompt, image, request_id):
    """Image (if any) + one question, no history. The request id goes first so nothing comes from cache."""
    first = f"Request id {request_id}."
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
    if (api_name, style) not in NO_USAGE:
        payload["stream_options"] = {"include_usage": True}      # token counts at the end
    dep = urllib.parse.quote(api_name, safe="")
    if style == "v1":
        url = f"{endpoint}/openai/v1/chat/completions"
        payload["model"] = api_name
    elif style == "v1-preview":
        url = f"{endpoint}/openai/v1/chat/completions?api-version=preview"
        payload["model"] = api_name
    elif style == "deploy-2025":
        url = f"{endpoint}/openai/deployments/{dep}/chat/completions?api-version=2025-04-01-preview"
    else:
        url = f"{endpoint}/openai/deployments/{dep}/chat/completions?api-version=2024-10-21"
    return url, payload, {"api-key": key}


def short_error(body):
    try:
        d = json.loads(body)
        err = d.get("error", d) if isinstance(d, dict) else d
        if isinstance(err, dict):
            return f"{err.get('code', '') or err.get('type', '')} {err.get('message', '')}".strip()[:300]
    except Exception:
        pass
    return (body or "").strip()[:300].replace("\n", " ")


def read_event(style, ev, r):
    """Returns the answer text in one streamed event; stores token counts in r."""
    if style == "anthropic":
        t = ev.get("type")
        if t == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            r["cached_tokens"] = u.get("cache_read_input_tokens") or 0
            r["input_tokens"] = (u.get("input_tokens") or 0) + r["cached_tokens"] + \
                (u.get("cache_creation_input_tokens") or 0)
        elif t == "message_delta":
            r["output_tokens"] = (ev.get("usage") or {}).get("output_tokens", r.get("output_tokens"))
        elif t == "content_block_delta":
            return (ev.get("delta") or {}).get("text") or ""
        return ""
    u = ev.get("usage")
    if isinstance(u, dict):
        r["input_tokens"] = u.get("prompt_tokens")
        r["output_tokens"] = u.get("completion_tokens")
        r["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        r["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    choices = ev.get("choices") or []
    return ((choices[0].get("delta") or {}).get("content") or "") if choices else ""


def ask(style, endpoint, key, api_name, prompt, image=None):
    """One request. Connection opened BEFORE the clock; clock = sending -> last word received."""
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image,
                                          uuid.uuid4().hex[:12])
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    r = dict(ok=False, status=None, error="", text="", pieces=[])
    conn = None
    try:
        t = time.perf_counter()
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=TIMEOUT,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=TIMEOUT)
        conn.connect()                                               # DNS + TCP + TLS
        r["setup"] = time.perf_counter() - t                         # logged, not in the times

        conn.putrequest("POST", (u.path or "/") + (f"?{u.query}" if u.query else ""),
                        skip_accept_encoding=True)
        for k, v in {**headers, "Content-Type": "application/json", "Accept": "text/event-stream",
                     "Content-Length": str(len(body))}.items():
            conn.putheader(k, v)
        t0 = time.perf_counter()                                     # ===== CLOCK STARTS =====
        conn.endheaders(body)
        resp = conn.getresponse()
        r["status"] = resp.status
        if resp.status >= 400:
            r["error"] = short_error(resp.read().decode("utf-8", "replace"))
            return r
        if "event-stream" not in (resp.getheader("Content-Type") or ""):  # server did not stream
            d = json.loads(resp.read().decode("utf-8", "replace"))
            r["total"] = time.perf_counter() - t0
            if style == "anthropic":
                r["text"] = "".join(c.get("text", "") for c in d.get("content", []))
            else:
                r["text"] = d["choices"][0]["message"]["content"] or ""
                read_event(style, {"usage": d.get("usage")}, r)
            r["pieces"], r["ok"] = [(r["total"], r["text"])], True
            return r
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            now = time.perf_counter() - t0
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
                r["pieces"].append((now, piece))                     # when each piece arrived
            if style == "anthropic" and ev.get("type") == "message_stop":
                break
        r["total"] = time.perf_counter() - t0                        # ===== CLOCK STOPS =====
        r["text"] = "".join(p for _, p in r["pieces"])
        r["ok"] = True
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    finally:
        if conn is not None:
            conn.close()


def call(style, endpoint, key, api_name, prompt, image=None):
    r = ask(style, endpoint, key, api_name, prompt, image)
    if not r["ok"] and r["status"] == 400 and "stream_options" in r["error"].lower():
        NO_USAGE.add((api_name, style))                              # retry without token counts
        r = ask(style, endpoint, key, api_name, prompt, image)
    return r


def find_models(endpoint, key, labels):
    print("\nFinding API names (a tiny 'hi' message each)...")
    found = {}
    for label, company, names in MODELS:
        if label not in labels:
            continue
        hit, errors = None, set()
        for name in names:
            for st in (STYLES_CLAUDE if company == "Claude" else STYLES_OPENAI):
                res = call(st, endpoint, key, name, "Reply with the single word: hi")
                if res["ok"]:
                    hit = {"api_name": name, "style": st}
                    break
                if res["status"] == 401:
                    sys.exit(f"\nThe key was rejected (401: {res['error']}). Copy the PRIMARY key again.")
                if res["status"] != 404:
                    errors.add(f"{res['status']}: {res['error'][:90]}")
            if hit:
                break
        if hit:
            found[label] = hit
            print(f"  OK        {label:<18} -> '{hit['api_name']}'")
        else:
            print(f"  NOT FOUND {label:<18} {'; '.join(sorted(errors))[:150]}")
    return found


# Grading (after the clock)
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def words(text):
    return [NUMBER_WORDS.get(w, w) for w in re.findall(r"[a-z]+|\d+", (text or "").lower())]


def grade(q, text):
    """Returns (verdict, deciding word). Looks in this order and uses the first answer word found:
    an explicit answer ("answer is X", "Answer: X", \\boxed{X}), bold text (**X**), the whole reply."""
    text = text or ""
    places = [m.group(1) for m in re.finditer(r"answer(?: would be| is)?\s*[:\-]?\s*([^\n.]{0,40})",
                                              text, re.I)]
    places += re.findall(r"\\boxed\{([^}]*)\}", text) + re.findall(r"\*\*([^*]+)\*\*", text) + [text]
    for place in places:
        for w in words(place):
            if w in q["accept"]:
                return "correct", w
            if w in q["wrong"]:
                return "wrong", w
    return "review", ""


def answer_time(pieces, word):
    """Seconds after sending when the deciding word had arrived."""
    if not word:
        return None
    target = words("".join(p for _, p in pieces)).index(word)
    text = ""
    for t, piece in pieces:
        text += piece
        w = words(text)
        if len(w) > target and w[target] == word:
            return t
    return None


# The test
def run(endpoint, key, found, image, rounds, no_image_rounds, pause):
    rows, labels = [], list(found)
    plan = [("no", r) for r in range(1, no_image_rounds + 1)] + [("yes", r) for r in range(1, rounds + 1)]
    for with_image, rnd in plan:
        questions = random.sample(QUESTIONS, len(QUESTIONS))
        for q in questions:
            prompt = f"{q['text']} {q['instr']}"
            for label in random.sample(labels, len(labels)):
                time.sleep(pause)
                r = call(found[label]["style"], endpoint, key, found[label]["api_name"], prompt,
                         image if with_image == "yes" else None)
                # ----- clock has stopped; bookkeeping only from here -----
                verdict, word = grade(q, r["text"]) if r["ok"] else ("error", "")
                if with_image == "no" and r["ok"] and len(words(r["text"])) > 3:
                    verdict, word = "no answer", ""      # e.g. "No image was provided": not an answer
                rows.append(dict(
                    round=rnd, image=with_image, model=label, q=q, prompt=prompt, reply=r["text"],
                    verdict=verdict, word=word,
                    followed="yes" if r["ok"] and len(words(r["text"])) <= 3 else "no",
                    first=r["pieces"][0][0] if r["pieces"] else None, answer=answer_time(r["pieces"], word),
                    total=r.get("total"), input_tokens=r.get("input_tokens"),
                    output_tokens=r.get("output_tokens"), reasoning_tokens=r.get("reasoning_tokens"),
                    cached_tokens=r.get("cached_tokens"), setup=r.get("setup"),
                    error="" if r["ok"] else f"{r['status']}: {r['error']}"))
        title = f"No-image check {rnd}/{no_image_rounds}" if with_image == "no" else f"Round {rnd}/{rounds}"
        print(f"\n{title}  ({datetime.now():%H:%M})")
        for label in labels:
            mine = [x for x in rows if x["round"] == rnd and x["image"] == with_image and x["model"] == label]
            missed = [x["q"]["id"] for x in sorted(mine, key=lambda x: int(x["q"]["id"][1:]))
                      if x["verdict"] != "correct"]
            right = sum(x["verdict"] == "correct" for x in mine)
            print(f"  {label:<18} {right:>2}/{len(mine)} correct   not correct: {' '.join(missed) or '-'}")
    return rows


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def model_stats(rows, label):
    mine = [x for x in rows if x["model"] == label and x["image"] == "yes"]
    ok = [x for x in mine if x["verdict"] != "error"]
    same, history_ok = [], True
    for q in QUESTIONS:
        these = [x for x in ok if x["q"]["id"] == q["id"]]
        answers = [x["word"] or x["reply"].strip().lower() for x in these]
        if answers:
            same.append(Counter(answers).most_common(1)[0][1] / len(answers))
        sizes = [x["input_tokens"] for x in these if x["input_tokens"]]
        if sizes and max(sizes) - min(sizes) > 20:        # same question must be the same size every round
            history_ok = False
    return dict(
        failed=len(mine) - len(ok),
        accuracy=sum(x["verdict"] == "correct" for x in ok) / len(ok) if ok else None,
        review=sum(x["verdict"] == "review" for x in ok),
        consistency=statistics.mean(same) if same else None,
        first=med([x["first"] for x in ok]), answer=med([x["answer"] for x in ok]),
        total=med([x["total"] for x in ok]),
        per_q={q["id"]: med([x["total"] for x in ok if x["q"]["id"] == q["id"]]) for q in QUESTIONS},
        input_tokens=med([x["input_tokens"] for x in ok]), output_tokens=med([x["output_tokens"] for x in ok]),
        reasoning_tokens=med([x["reasoning_tokens"] for x in ok]), setup=med([x["setup"] for x in ok]),
        cached=max((x["cached_tokens"] or 0 for x in ok), default=0),
        history="OK" if history_ok else "CHECK: input size changed between rounds")


# Excel
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def write_excel(path, rows, labels, image_name, rounds, no_image_rounds):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    bold, fill = Font(bold=True), PatternFill("solid", fgColor="FFFF00")
    wb = Workbook()

    def header(ws, cells, widths):
        ws.append(cells)
        for c in ws[ws.max_row]:
            c.font, c.fill = bold, fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = f"B{ws.max_row + 1}"

    # Answers. The Results formulas use: B model, C question, D category, E with image, I verdict, J format.
    wa = wb.active
    wa.title = "Answers"
    header(wa, ["Round", "Model", "Q", "Category", "With image", "Question sent", "Correct answer", "Reply",
                "Verdict", "Followed format", "First word (s)", "Answer word (s)", "Total (s)",
                "Input tokens", "Output tokens", "Reasoning tokens", "Cached tokens",
                "Network setup (s, not in times)", "Error"],
           [7, 18, 5, 15, 8, 45, 18, 45, 9, 9, 10, 10, 10, 9, 9, 9, 9, 12, 30])
    for x in sorted(rows, key=lambda x: (x["image"], x["round"], x["model"], int(x["q"]["id"][1:]))):
        q = x["q"]
        rnd = lambda v: None if v is None else round(v, 3)
        wa.append([ILLEGAL.sub("", v) if isinstance(v, str) else v for v in [
            x["round"], x["model"], q["id"], q["category"], x["image"], x["prompt"], q["answer"], x["reply"],
            x["verdict"], x["followed"], rnd(x["first"]), rnd(x["answer"]), rnd(x["total"]),
            x["input_tokens"], x["output_tokens"], x["reasoning_tokens"], x["cached_tokens"],
            rnd(x["setup"]), x["error"]]])

    # Results: one column per model.
    wr = wb.create_sheet("Results", 0)
    wr.append([f"Image {image_name} | {rounds} rounds with the image, {no_image_rounds} without | accuracy "
               f"updates when you change 'review' verdicts in the Answers sheet"])
    wr["A1"].font = bold
    header(wr, ["Measure"] + labels, [48] + [18] * len(labels))
    top = wr.max_row

    def acc(extra="", image="yes"):
        def f(col):
            m = f'Answers!$B:$B,{col}${top},Answers!$E:$E,"{image}"{extra}'
            return (f'=IFERROR(COUNTIFS({m},Answers!$I:$I,"correct")/'
                    f'(COUNTIFS({m})-COUNTIFS({m},Answers!$I:$I,"error")),"")')
        return f

    def add(label, make, pct=False):
        wr.append([label] + [make(get_column_letter(i + 2)) if callable(make) else make[i]
                             for i in range(len(labels))])
        if pct:
            for c in wr[wr.max_row][1:]:
                c.number_format = "0%"

    def section(title):
        wr.append([title])
        wr.cell(wr.max_row, 1).font = bold

    stats = [model_stats(rows, l) for l in labels]
    section("ACCURACY (with the image)")
    add("All 24 questions", acc(), True)
    add("Replies to review (count as wrong until marked)",
        lambda c: f'=COUNTIFS(Answers!$B:$B,{c}${top},Answers!$E:$E,"yes",Answers!$I:$I,"review")')
    add("Followed the answer instruction",
        lambda c: f'=IFERROR(COUNTIFS(Answers!$B:$B,{c}${top},Answers!$E:$E,"yes",Answers!$J:$J,"yes")/'
                  f'COUNTIFS(Answers!$B:$B,{c}${top},Answers!$E:$E,"yes"),"")', True)
    add("Same answer every round", [s["consistency"] for s in stats], True)
    section("ACCURACY BY CATEGORY")
    for cat in dict.fromkeys(q["category"] for q in QUESTIONS):
        add(f"  {cat}", acc(f',Answers!$D:$D,"{cat}"'), True)
    section("ACCURACY BY QUESTION")
    for q in QUESTIONS:
        add(f"  {q['id']} {q['text']}", acc(f',Answers!$C:$C,"{q["id"]}"'), True)
    section("TIME (median seconds from sending the request)")
    add("  First word", [s["first"] for s in stats])
    add("  Answer word", [s["answer"] for s in stats])
    add("  Total (last word)", [s["total"] for s in stats])
    for q in QUESTIONS:
        add(f"  Total, {q['id']}", [s["per_q"][q["id"]] for s in stats])
    section("TOKENS (median)")
    add("  Input (image + question)", [s["input_tokens"] for s in stats])
    add("  Output", [s["output_tokens"] for s in stats])
    add("  Reasoning (thinking models only)", [s["reasoning_tokens"] for s in stats])
    if no_image_rounds:
        section("NO-IMAGE CHECK: correct WITHOUT the image (high = answerable without looking)")
        add("  All 24 questions", acc(image="no"), True)
        for q in QUESTIONS:
            add(f"  {q['id']} {q['text']}", acc(f',Answers!$C:$C,"{q["id"]}"', image="no"), True)
    section("CHECKS")
    add("  Network setup, not in the times (s)", [s["setup"] for s in stats])
    add("  Cached tokens (should be 0)", [s["cached"] for s in stats])
    add("  No history between questions", [s["history"] for s in stats])
    add("  Failed requests", [s["failed"] for s in stats])

    # Questions.
    wq = wb.create_sheet("Questions")
    header(wq, ["Q", "Question", "Instruction added", "Category", "Correct answer", "Counted correct",
                "Counted wrong"], [5, 50, 32, 16, 28, 40, 40])
    for q in QUESTIONS:
        wq.append([q["id"], q["text"], q["instr"], q["category"], q["answer"], ", ".join(sorted(q["accept"])),
                   ", ".join(sorted(q["wrong"], key=lambda w: (len(w), w)))])
    wb.save(path)
    return stats


# Main
def main():
    ap = argparse.ArgumentParser(description="1 image, 24 questions, several AI models")
    ap.add_argument("--models", default="all", help='"all" or e.g. "GPT-4o,Llama-4 Maverick"')
    ap.add_argument("--image", default=IMAGE, help="image name in the images folder, or a file path")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--no-image-rounds", type=int, default=NO_IMAGE_ROUNDS)
    ap.add_argument("--pause", type=float, default=PAUSE, help="seconds between requests")
    ap.add_argument("--recheck", action="store_true", help="find the API names again")
    args = ap.parse_args()

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("This needs openpyxl. Install it with:  pip3 install openpyxl")
    all_labels = [m[0] for m in MODELS]
    if args.models.strip().lower() == "all":
        labels = all_labels
    else:
        lookup = {l.lower().replace(" ", ""): l for l in all_labels}
        labels = [lookup.get(w.strip().lower().replace(" ", "")) for w in args.models.split(",")]
        if None in labels:
            sys.exit(f"Unknown model in '{args.models}'. Use: {', '.join(all_labels)}")
    image_name, image = load_image(args.image)

    cfg = load_config()
    endpoint = get_endpoint(cfg)
    key = get_key()
    found = cfg.get("found", {})
    missing = [l for l in labels if args.recheck or l not in found]
    if missing:
        for l in missing:
            found.pop(l, None)
        found.update(find_models(endpoint, key, missing))
        cfg["found"] = {l: v for l, v in found.items() if l in all_labels}
        save_config(cfg)
    found = {l: found[l] for l in labels if l in found}
    if not found:
        sys.exit("No model answered, so nothing was run.")
    skipped = [l for l in labels if l not in found]
    if skipped:
        print(f"Skipping (not found): {', '.join(skipped)}")

    rounds, blind = max(1, args.rounds), max(0, args.no_image_rounds)
    n = (rounds + blind) * len(QUESTIONS) * len(found)
    print(f"\n{image_name} | {len(found)} models x {len(QUESTIONS)} questions x ({rounds} rounds + {blind} "
          f"without image) = {n} requests, about {n * (args.pause + 1.8) / 3600:.1f} h")
    rows = run(endpoint, key, found, image, rounds, blind, args.pause)

    path = HERE / f"results_{datetime.now():%Y-%m-%d_%H%M}.xlsx"
    stats = write_excel(path, rows, list(found), image_name, rounds, blind)
    print(f"\n{'Model':<20}{'Accuracy':>9}{'Review':>8}{'Same':>7}{'1st word':>10}{'Answer':>9}{'Total':>8}")
    for label, s in zip(found, stats):
        print(f"{label:<20}{(s['accuracy'] or 0):>9.0%}{s['review']:>8}{(s['consistency'] or 0):>7.0%}"
              f"{s['first'] or 0:>9.2f}s{s['answer'] or 0:>8.2f}s{s['total'] or 0:>7.2f}s")
    for label, s in zip(found, stats):
        if s["cached"] or s["history"] != "OK":
            print(f"WARNING {label}: cached tokens {s['cached']}, history check: {s['history']}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
