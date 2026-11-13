#!/usr/bin/env python3
# Experiment 05: ten questions about one photo, all six models, several rounds.
# Run: python3 test_05.py --rounds 10    (add --natural to drop the answer instructions)
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
IMAGE = "IMG_1"                                      # name in IMAGES_DIR, or a full file path
ROUNDS = 10
PAUSE = 1.0                                          # seconds between requests
TIMEOUT = 180

YES_NO = "Answer with only yes or no."
ONE_WORD = "Answer with one word."
NUMBER = "Answer with a number."
COLORS = {"black", "white", "brown", "red", "blue", "green", "yellow", "orange", "gold", "golden",
          "grey", "gray", "silver", "pink", "purple", "tan", "beige", "leather"}
NUMBERS = {str(n) for n in range(21)}

# Each question: id, text, instruction, ability, steps, premise, correct answer (shown in Excel),
# accepted words (= correct), wrong words (= wrong). Anything else is marked "review".
QUESTIONS = [
    dict(id="Q1", text="What animal is in this image?", instr=ONE_WORD, ability="Recognition",
         steps=2, premise="valid", answer="jaguar", accept={"jaguar", "jaguars"},
         wrong={"leopard", "leopards", "cheetah", "cheetahs", "tiger", "tigers", "lion", "lions",
                "puma", "cougar", "lynx", "ocelot"}),
    dict(id="Q2", text="Is there a leopard in this image?", instr=YES_NO, ability="Recognition",
         steps=2, premise="false premise", answer="no", accept={"no"}, wrong={"yes"}),
    dict(id="Q3", text="How many jaguars are in this image?", instr=NUMBER, ability="Counting",
         steps=3, premise="valid", answer="2", accept={"2"}, wrong=NUMBERS - {"2"}),
    dict(id="Q4", text="How many eyes are visible in this image?", instr=NUMBER, ability="Counting",
         steps=4, premise="valid", answer="3", accept={"3"}, wrong=NUMBERS - {"3"}),
    dict(id="Q5", text="How many paws are visible in this image?", instr=NUMBER, ability="Counting",
         steps=4, premise="valid", answer="0", accept={"0", "none", "no"}, wrong=NUMBERS - {"0"}),
    dict(id="Q6", text="What is above the jaguars?", instr=ONE_WORD, ability="Position",
         steps=3, premise="valid", answer="branch / tree / log",
         accept={"branch", "branches", "tree", "trees", "log", "logs", "wood", "wooden", "trunk", "bark",
                 "limb"},
         wrong={"rock", "rocks", "stone", "stones", "sky", "nothing", "none", "ceiling"}),
    dict(id="Q7", text="Are the jaguars awake?", instr=YES_NO, ability="Reasoning",
         steps=4, premise="valid", answer="yes", accept={"yes", "awake"},
         wrong={"no", "asleep", "sleeping"}),
    dict(id="Q8", text="What color is the jaguar's collar?", instr=ONE_WORD, ability="Hallucination",
         steps=3, premise="false premise", answer="none (there is no collar)",
         accept={"none", "no", "not", "nothing", "neither"}, wrong=COLORS),
    dict(id="Q9", text="Is the cat in this image white or black?", instr=ONE_WORD,
         ability="Hallucination", steps=3, premise="false premise",
         answer="neither (it is golden with spots)",
         accept={"neither", "none", "golden", "gold", "yellow", "orange", "tan", "tawny", "brown",
                 "spotted"},
         wrong={"white", "black", "both"}),
    dict(id="Q10", text="Is there a jaguar and a tiger in this image?", instr=YES_NO, ability="Logic",
         steps=4, premise="false premise", answer="no", accept={"no"}, wrong={"yes"}),
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
    """Image + one question, no history. The request id goes first so nothing is reused from cache."""
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
    """Returns the text piece in one streamed event; stores token counts in r."""
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
            d = ev.get("delta") or {}
            return d.get("text") or ""                               # thinking text is not answer text
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
    """One request. Connection is opened BEFORE the clock starts.
    Clock: start = sending image + question, stop = last word received."""
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image,
                                          uuid.uuid4().hex[:12])
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    https = u.scheme == "https"
    port = u.port or (443 if https else 80)
    r = dict(ok=False, status=None, error="", text="", pieces=[])
    conn = None
    try:
        t = time.perf_counter()                                      # network setup (not in times)
        if https:
            conn = http.client.HTTPSConnection(u.hostname, port, timeout=TIMEOUT,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, port, timeout=TIMEOUT)
        conn.connect()                                               # DNS + TCP + TLS
        r["setup"] = time.perf_counter() - t

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
            raw = resp.read().decode("utf-8", "replace")
            r["total"] = time.perf_counter() - t0
            d = json.loads(raw)
            if style == "anthropic":
                r["text"] = "".join(c.get("text", "") for c in d.get("content", []))
            else:
                r["text"] = d["choices"][0]["message"]["content"] or ""
                read_event(style, {"usage": d.get("usage")}, r)
            r["pieces"] = [(r["total"], r["text"])]
            r["ok"] = True
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
    """Returns (verdict, deciding word). Long replies often explain first and answer later
    ("two jaguars ... Answer: 3"), so look in this order and use the first answer-word found:
      1. an explicit final answer: "answer is X", "Answer: X", \\boxed{X}
      2. bold text (**X**)
      3. the whole reply"""
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
    """When the deciding word had fully arrived (seconds after sending)."""
    if not word:
        return None
    target = words("".join(p for _, p in pieces)).index(word)     # its position in the reply
    text = ""
    for t, piece in pieces:
        text += piece
        w = words(text)
        if len(w) > target and w[target] == word:
            return t
    return None


# The experiment
def run(endpoint, key, found, image, rounds, natural, pause):
    rows = []
    labels = list(found)
    for rnd in range(1, rounds + 1):
        questions = QUESTIONS[:]
        random.shuffle(questions)
        for q in questions:
            prompt = q["text"] if natural else f"{q['text']} {q['instr']}"
            for label in random.sample(labels, len(labels)):
                time.sleep(pause)
                r = call(found[label]["style"], endpoint, key, found[label]["api_name"], prompt, image)
                # ----- clock has stopped; bookkeeping only from here -----
                verdict, word = grade(q, r["text"]) if r["ok"] else ("error", "")
                first = r["pieces"][0][0] if r["pieces"] else None
                rows.append(dict(
                    round=rnd, model=label, q=q, prompt=prompt, reply=r["text"], verdict=verdict,
                    word=word, followed="yes" if r["ok"] and len(words(r["text"])) <= 3 else "no",
                    first=first, answer=answer_time(r["pieces"], word), total=r.get("total"),
                    input_tokens=r.get("input_tokens"), output_tokens=r.get("output_tokens"),
                    reasoning_tokens=r.get("reasoning_tokens"), cached_tokens=r.get("cached_tokens"),
                    setup=r.get("setup"), error="" if r["ok"] else f"{r['status']}: {r['error']}"))
        print(f"\nRound {rnd}/{rounds}")
        mark = {"correct": "ok", "wrong": "x", "review": "?", "error": "ERR"}
        for label in labels:
            done = {x["q"]["id"]: x for x in rows if x["round"] == rnd and x["model"] == label}
            print(f"  {label:<18}" + "  ".join(
                f"{q['id']} {mark[done[q['id']]['verdict']]} {done[q['id']]['total'] or 0:.1f}s"
                for q in QUESTIONS))
    return rows


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def model_stats(rows, label):
    mine = [x for x in rows if x["model"] == label]
    ok = [x for x in mine if x["verdict"] != "error"]
    same = []                                          # consistency: share of the usual answer
    for q in QUESTIONS:
        ans = [x["word"] or x["reply"].strip().lower() for x in ok if x["q"]["id"] == q["id"]]
        if ans:
            same.append(Counter(ans).most_common(1)[0][1] / len(ans))
    history_ok = True                                  # same question = same input size every round
    for q in QUESTIONS:
        sizes = [x["input_tokens"] for x in ok if x["q"]["id"] == q["id"] and x["input_tokens"]]
        if sizes and max(sizes) - min(sizes) > 20:
            history_ok = False
    return dict(
        requests=len(mine), failed=len(mine) - len(ok),
        accuracy=sum(x["verdict"] == "correct" for x in ok) / len(ok) if ok else None,
        review=sum(x["verdict"] == "review" for x in ok),
        consistency=statistics.mean(same) if same else None,
        first=med([x["first"] for x in ok]), answer=med([x["answer"] for x in ok]),
        total=med([x["total"] for x in ok]),
        per_q={q["id"]: med([x["total"] for x in ok if x["q"]["id"] == q["id"]]) for q in QUESTIONS},
        input_tokens=med([x["input_tokens"] for x in ok]), output_tokens=med([x["output_tokens"] for x in ok]),
        reasoning_tokens=med([x["reasoning_tokens"] for x in ok]), setup=med([x["setup"] for x in ok]),
        cached=max((x["cached_tokens"] or 0 for x in ok), default=0),
        history="OK" if history_ok else "CHECK: input tokens changed between rounds")


# Excel
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def write_excel(path, rows, labels, image_name, rounds, natural):
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

    # --- Answers (columns are referenced by the Results formulas: B model, D ability, E steps,
    #     F premise, C question id, J verdict, K followed format) ---
    wa = wb.active
    wa.title = "Answers"
    header(wa, ["Round", "Model", "Q", "Ability", "Steps", "Premise", "Question sent", "Correct answer",
                "Reply", "Verdict", "Followed format", "First word (s)", "Answer word (s)", "Total (s)",
                "Input tokens", "Output tokens", "Reasoning tokens", "Cached tokens",
                "Network setup (s, not in times)", "Error"],
           [7, 18, 5, 14, 6, 13, 45, 18, 45, 9, 9, 10, 10, 10, 9, 9, 9, 9, 12, 30])
    for x in sorted(rows, key=lambda x: (x["round"], x["model"], int(x["q"]["id"][1:]))):
        q = x["q"]
        wa.append([ILLEGAL.sub("", v) if isinstance(v, str) else v for v in [
            x["round"], x["model"], q["id"], q["ability"], q["steps"], q["premise"], x["prompt"],
            q["answer"], x["reply"], x["verdict"], x["followed"], x["first"] and round(x["first"], 3),
            x["answer"] and round(x["answer"], 3), x["total"] and round(x["total"], 3), x["input_tokens"],
            x["output_tokens"], x["reasoning_tokens"], x["cached_tokens"],
            x["setup"] and round(x["setup"], 3), x["error"]]])

    # --- Results: one column per model ---
    wr = wb.create_sheet("Results", 0)
    wr.append([f"Image {image_name} | {rounds} rounds | "
               f"{'natural questions (no answer instructions)' if natural else 'with answer instructions'}"
               f" | accuracy cells update when you change 'review' verdicts in the Answers sheet"])
    wr["A1"].font = bold
    header(wr, ["Measure"] + labels, [44] + [18] * len(labels))
    top = wr.max_row                                   # row holding the model names

    def acc(extra=""):
        """Accuracy formula for the model in this column (errors left out)."""
        def f(col):
            m = f'Answers!$B:$B,{col}${top}{extra}'
            return (f'=IFERROR(COUNTIFS({m},Answers!$J:$J,"correct")/'
                    f'(COUNTIFS({m})-COUNTIFS({m},Answers!$J:$J,"error")),"")')
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
    section("ACCURACY")
    add("All questions", acc(), True)
    add("Replies to review (count as wrong until marked)",
        lambda c: f'=COUNTIFS(Answers!$B:$B,{c}${top},Answers!$J:$J,"review")')
    add("Followed the answer format",
        lambda c: f'=IFERROR(COUNTIFS(Answers!$B:$B,{c}${top},Answers!$K:$K,"yes")/'
                  f'COUNTIFS(Answers!$B:$B,{c}${top}),"")', True)
    add("Consistency (same answer every round)", [s["consistency"] for s in stats], True)
    section("ACCURACY BY ABILITY")
    for ab in dict.fromkeys(q["ability"] for q in QUESTIONS):
        add(f"  {ab}", acc(f',Answers!$D:$D,"{ab}"'), True)
    section("ACCURACY BY REASONING STEPS")
    for st in sorted({q["steps"] for q in QUESTIONS}):
        add(f"  {st} steps", acc(f",Answers!$E:$E,{st}"), True)
    section("ACCURACY BY PREMISE")
    for p in ("valid", "false premise"):
        add(f"  {p}", acc(f',Answers!$F:$F,"{p}"'), True)
    section("ACCURACY BY QUESTION")
    for q in QUESTIONS:
        add(f"  {q['id']} {q['text']}", acc(f',Answers!$C:$C,"{q["id"]}"'), True)
    section("TIME (median seconds, from sending to arrival)")
    add("  First word", [s["first"] for s in stats])
    add("  Answer word", [s["answer"] for s in stats])
    add("  Total (last word)", [s["total"] for s in stats])
    for q in QUESTIONS:
        add(f"  Total, {q['id']}", [s["per_q"][q["id"]] for s in stats])
    section("TOKENS (median)")
    add("  Input (image + question)", [s["input_tokens"] for s in stats])
    add("  Output", [s["output_tokens"] for s in stats])
    add("  Reasoning (thinking models only)", [s["reasoning_tokens"] for s in stats])
    section("CHECKS")
    add("  Network setup, not in the times (s)", [s["setup"] for s in stats])
    add("  Cached tokens (should be 0)", [s["cached"] for s in stats])
    add("  No history between questions", [s["history"] for s in stats])
    add("  Failed requests", [s["failed"] for s in stats])

    # --- Questions ---
    wq = wb.create_sheet("Questions")
    header(wq, ["Q", "Question", "Instruction added", "Ability", "Steps", "Premise", "Correct answer",
                "Counted correct", "Counted wrong"], [5, 45, 28, 14, 6, 13, 26, 40, 40])
    for q in QUESTIONS:
        wq.append([q["id"], q["text"], "" if natural else q["instr"], q["ability"], q["steps"],
                   q["premise"], q["answer"], ", ".join(sorted(q["accept"])),
                   ", ".join(sorted(q["wrong"], key=lambda w: (len(w), w)))])
    wb.save(path)
    return stats


# Main
def main():
    ap = argparse.ArgumentParser(description="1 image, 10 questions, several models")
    ap.add_argument("--models", default="all", help='"all" or e.g. "GPT-4o,Llama-4 Maverick"')
    ap.add_argument("--image", default=IMAGE, help="image name in the images folder, or a file path")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--pause", type=float, default=PAUSE, help="seconds between requests")
    ap.add_argument("--natural", action="store_true", help="leave out the answer instructions")
    ap.add_argument("--recheck", action="store_true", help="find the API names again")
    args = ap.parse_args()

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("This needs openpyxl for the Excel file. Install it with:  pip3 install openpyxl")
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

    rounds = max(1, args.rounds)
    n = rounds * len(QUESTIONS) * len(found)
    print(f"\n{image_name} | {len(found)} models x {len(QUESTIONS)} questions x {rounds} rounds = "
          f"{n} requests (about {n * (args.pause + 1.5) / 60:.0f} min)")
    rows = run(endpoint, key, found, image, rounds, args.natural, args.pause)

    path = HERE / f"results_{datetime.now():%Y-%m-%d_%H%M}{'_natural' if args.natural else ''}.xlsx"
    stats = write_excel(path, rows, list(found), image_name, rounds, args.natural)
    print(f"\n{'Model':<20}{'Accuracy':>9}{'Review':>8}{'Same':>7}{'1st word':>10}{'Answer':>9}{'Total':>8}")
    for label, s in zip(found, stats):
        print(f"{label:<20}{(s['accuracy'] or 0):>9.0%}{s['review']:>8}{(s['consistency'] or 0):>7.0%}"
              f"{s['first'] or 0:>9.2f}s{s['answer'] or 0:>8.2f}s{s['total'] or 0:>7.2f}s")
    print("(times are medians in seconds from sending the request; 'review' replies count as wrong "
          "until you mark them)")
    for label, s in zip(found, stats):
        if s["cached"] or s["history"] != "OK":
            print(f"WARNING {label}: cached tokens {s['cached']}, history check: {s['history']}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
