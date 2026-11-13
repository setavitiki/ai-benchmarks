#!/usr/bin/env python3
# Experiment 04: ten yes/no questions about one photo, in the style of the MME benchmark.
# Run: python3 test_04.py --model "Llama-4 Maverick" --rounds 10
import argparse
import base64
import getpass
import http.client
import json
import os
import random
import re
import shutil
import socket
import ssl
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

# Settings
IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "test_images"
HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "results_mme.xlsx"
CONFIG_FILE = HERE / "patriot_api_config.json"        # endpoint + found API names (never the key)

IMAGE = "IMG_1"                                       # name in IMAGES_DIR, or a full file path
SUFFIX = " Please answer yes or no."                  # MME's exact instruction

# (id, MME category, MME group, question, correct answer)
# Pairs share a category: one "yes" question and one "no" question.
# Harder variant of Q2: change "tiger" to "leopard" (jaguars and leopards look very alike).
QUESTIONS = [
    ("Q1", "Existence", "Perception", "Is there a jaguar in this image?", "yes"),
    ("Q2", "Existence", "Perception", "Is there a tiger in this image?", "no"),
    ("Q3", "Count", "Perception", "Are there two jaguars in this image?", "yes"),
    ("Q4", "Count", "Perception", "Are there three jaguars in this image?", "no"),
    ("Q5", "Position", "Perception", "Is the fallen tree trunk above the jaguars in this image?", "yes"),
    ("Q6", "Position", "Perception", "Is the fallen tree trunk below the jaguars in this image?", "no"),
    ("Q7", "Color", "Perception", "Is the jaguars' fur mainly golden-yellow in this image?", "yes"),
    ("Q8", "Color", "Perception", "Is the jaguars' fur mainly grey in this image?", "no"),
    ("Q9", "Commonsense reasoning", "Cognition",
     "Should a visitor keep a safe distance from these animals?", "yes"),
    ("Q10", "Commonsense reasoning", "Cognition",
     "Is it safe for a child to walk up and pet these animals?", "no"),
]

MODELS = [
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

ROUNDS = 10                   # each round asks all 10 questions, in a new random order
TIMEOUT = 180
PAUSE = 1.0                   # seconds between requests
MAX_IMAGE_BYTES = 3_500_000
ANTHROPIC_VERSION = "2023-06-01"

STYLES_OPENAI = ["v1", "v1-preview", "deploy-2025", "deploy-2024"]
STYLES_CLAUDE = ["anthropic"] + STYLES_OPENAI
TIMING_HEADER = re.compile(r"(time|ms\b|-ms|duration|latency|processing|elapsed)", re.I)


# Config / inputs
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


MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif"}


def find_image(name):
    p = Path(name).expanduser()
    if p.is_file():
        return p
    if IMAGES_DIR.is_dir():
        for f in sorted(IMAGES_DIR.iterdir()):
            if f.is_file() and f.stem.strip().upper() == name.strip().upper():
                return f
    return None


def load_image(path):
    suffix = path.suffix.lower()
    if suffix in MIME and path.stat().st_size <= MAX_IMAGE_BYTES:
        return MIME[suffix], base64.b64encode(path.read_bytes()).decode()
    if shutil.which("sips"):
        out = Path(tempfile.mkdtemp()) / (path.stem + ".jpg")
        cmd = ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "85"]
        if path.stat().st_size > MAX_IMAGE_BYTES:
            cmd += ["-Z", "2048"]
        subprocess.run(cmd + [str(path), "--out", str(out)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "image/jpeg", base64.b64encode(out.read_bytes()).decode()
    if suffix in MIME:
        return MIME[suffix], base64.b64encode(path.read_bytes()).decode()
    raise RuntimeError(f"{path.name}: unsupported format; save it as .jpg or .png")


# Request building
QUIRKS = {}   # per (api_name, style): does the server accept stream_options?


def build_request(style, endpoint, key, api_name, prompt, image, nonce, quirks):
    tag = f"Request id {nonce}." if nonce else None      # unique start = no cached shortcut
    if style == "anthropic":
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64",
                                                           "media_type": image[0], "data": image[1]}})
        payload = {"model": api_name, "max_tokens": 1024, "stream": True,
                   "messages": [{"role": "user", "content": content}]}
        if tag:
            payload["system"] = tag
        return (f"{endpoint}/anthropic/v1/messages", payload,
                {"api-key": key, "x-api-key": key, "anthropic-version": ANTHROPIC_VERSION})

    content = prompt if not image else [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{image[0]};base64,{image[1]}"}}]
    messages = ([{"role": "system", "content": tag}] if tag else []) + \
        [{"role": "user", "content": content}]
    payload = {"messages": messages, "stream": True}
    if quirks.get("usage", True):
        payload["stream_options"] = {"include_usage": True}   # token counts at the end
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
            msg = f"{err.get('code', '') or err.get('type', '')} {err.get('message', '')}".strip()
            if msg:
                return msg[:300]
    except Exception:
        pass
    return (body or "").strip()[:300].replace("\n", " ")


def parse_event(style, ev, acc):
    """One streamed event -> ('answer'|'thinking'|None, text). Token counts go into acc."""
    if not isinstance(ev, dict):
        return None, ""
    if style == "anthropic":
        t = ev.get("type")
        if t == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            acc["input_tokens"] = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
                + (u.get("cache_creation_input_tokens") or 0)
            acc["cached_tokens"] = u.get("cache_read_input_tokens") or 0
        elif t == "message_delta":
            u = ev.get("usage") or {}
            if "output_tokens" in u:
                acc["output_tokens"] = u["output_tokens"]
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") == "thinking_delta":
                return "thinking", d.get("thinking") or ""
            return "answer", d.get("text") or ""
        return None, ""

    u = ev.get("usage")
    if isinstance(u, dict):
        acc["input_tokens"] = u.get("prompt_tokens", acc.get("input_tokens"))
        acc["output_tokens"] = u.get("completion_tokens", acc.get("output_tokens"))
        pdet = u.get("prompt_tokens_details") or {}
        if pdet.get("cached_tokens") is not None:
            acc["cached_tokens"] = pdet["cached_tokens"]
        det = u.get("completion_tokens_details") or {}
        if det.get("reasoning_tokens") is not None:
            acc["reasoning_tokens"] = det["reasoning_tokens"]
    choices = ev.get("choices") or []
    if not choices:
        return None, ""
    d = choices[0].get("delta") or {}
    thinking = d.get("reasoning_content") or d.get("reasoning") or ""
    if thinking and isinstance(thinking, str):
        return "thinking", thinking
    return "answer", d.get("content") or ""


# Upload measurement
def bytes_not_yet_acknowledged(sock):
    """Bytes sent that the server has NOT yet confirmed receiving (None = can't tell)."""
    try:
        if sys.platform == "darwin":
            return sock.getsockopt(socket.SOL_SOCKET, 0x1024)          # SO_NWRITE
        if sys.platform.startswith("linux"):
            import fcntl
            import termios
            buf = fcntl.ioctl(sock.fileno(), termios.TIOCOUTQ, b"\0\0\0\0")
            return struct.unpack("i", buf)[0]
    except Exception:
        pass
    return None


def wait_until_acknowledged(sock, t0, limit=60):
    if bytes_not_yet_acknowledged(sock) is None:
        return None
    deadline = time.perf_counter() + limit
    while time.perf_counter() < deadline:
        if not bytes_not_yet_acknowledged(sock):
            return time.perf_counter() - t0
        time.sleep(0.0005)
    return None


# One timed request
def measure(style, endpoint, key, api_name, prompt, image, nonce):
    quirks = QUIRKS.setdefault((api_name, style), {})
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image, nonce, quirks)
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    host, https = u.hostname, u.scheme == "https"
    port = u.port or (443 if https else 80)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    r = dict(ok=False, status=None, error="", text="", bytes_sent=len(body), server_timing="")
    acc, conn = {}, None
    try:
        t0 = time.perf_counter()                                   # clock starts
        addr = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4]
        r["dns"] = time.perf_counter() - t0
        t = time.perf_counter()
        sock = socket.create_connection(addr[:2], timeout=TIMEOUT)
        r["tcp"] = time.perf_counter() - t                         # = 1 network round trip
        t = time.perf_counter()
        if https:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        r["tls"] = time.perf_counter() - t

        conn = (http.client.HTTPSConnection if https else http.client.HTTPConnection)(
            host, port, timeout=TIMEOUT)
        conn.sock = sock
        conn.putrequest("POST", path, skip_accept_encoding=True)
        for k, v in {**headers, "Content-Type": "application/json", "Accept": "text/event-stream",
                     "Content-Length": str(len(body)), "User-Agent": "patriot-mme/1.0"}.items():
            conn.putheader(k, v)
        send_start = time.perf_counter() - t0
        conn.endheaders(body)
        acked = wait_until_acknowledged(sock, t0)
        r["upload"] = (acked - send_start) if acked is not None else None

        resp = conn.getresponse()
        r["status"] = resp.status
        r["server_timing"] = "; ".join(f"{k}={v}" for k, v in resp.getheaders()
                                       if TIMING_HEADER.search(k) and k.lower() != "date")
        if resp.status >= 400:
            r["error"] = short_error(resp.read().decode("utf-8", "replace"))
            return r

        if "event-stream" not in (resp.getheader("Content-Type") or ""):
            raw = resp.read().decode("utf-8", "replace")
            r["total"] = r["first_token"] = time.perf_counter() - t0
            try:
                d = json.loads(raw)
                if style == "anthropic":
                    r["text"] = "".join(c.get("text", "") for c in d.get("content", []))
                    us = d.get("usage") or {}
                    acc.update(input_tokens=us.get("input_tokens"), output_tokens=us.get("output_tokens"))
                else:
                    r["text"] = d["choices"][0]["message"]["content"] or ""
                    parse_event(style, {"usage": d.get("usage")}, acc)
            except Exception:
                r["text"] = raw
            r.update(acc, ok=True)
            return r

        answer = []
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            now = time.perf_counter() - t0
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if isinstance(ev, dict) and (ev.get("type") == "error" or
                                         (ev.get("error") and not ev.get("choices"))):
                r["error"] = short_error(data)
                return r
            kind, piece = parse_event(style, ev, acc)
            if piece:
                r.setdefault("first_token", now)
                if kind == "answer":
                    answer.append(piece)
            if style == "anthropic" and ev.get("type") == "message_stop":
                break
        r["total"] = time.perf_counter() - t0                     # clock stops
        r.update(acc, ok=True, text="".join(answer))
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    finally:
        if conn is not None:
            conn.close()


def timed_call(style, endpoint, key, api_name, prompt, image=None, nonce=None):
    """measure(), retrying once without stream_options if the server rejects that field."""
    quirks = QUIRKS.setdefault((api_name, style), {})
    r = measure(style, endpoint, key, api_name, prompt, image, nonce)
    if not r["ok"] and r["status"] == 400 and "stream_options" in (r["error"] or "").lower() \
            and quirks.get("usage", True):
        quirks["usage"] = False
        r = measure(style, endpoint, key, api_name, prompt, image, nonce)
    return r


# Finding API names
def find_models(endpoint, key, labels):
    print("\nChecking which models answer (a tiny 'hi' message each)...")
    found = {}
    for label, company, names in MODELS:
        if label not in labels:
            continue
        styles = STYLES_CLAUDE if company == "Claude" else STYLES_OPENAI
        hit, errors = None, set()
        for name in names:
            for st in styles:
                res = timed_call(st, endpoint, key, name, "Reply with the single word: hi")
                if res["ok"]:
                    hit = {"api_name": name, "style": st}
                    break
                if res["status"] == 401:
                    sys.exit(f"\nThe key was rejected (401: {res['error']}).\n"
                             "Copy the PRIMARY key again from the API Projects page and rerun.")
                if res["status"] != 404:
                    errors.add(f"{res['status']}: {res['error'][:90]}")
            if hit:
                break
        if hit:
            found[label] = hit
            print(f"  OK        {label:<18} -> API name '{hit['api_name']}'")
        else:
            extra = f"  ({'; '.join(sorted(errors))[:150]})" if errors else ""
            print(f"  NOT FOUND {label:<18}{extra}")
    return found


# Scoring (after the clock stops)
def read_yes_no(text):
    """MME rule: the reply must START with yes or no. Returns 'yes', 'no' or 'invalid'."""
    t = re.sub(r"^[\s\"'*_`#>\-\.:]+", "", (text or "").lower())
    m = re.match(r"(yes|no)\b", t)
    return m.group(1) if m else "invalid"


# The experiment
def run_rounds(endpoint, key, api_name, style, image, rounds, use_nonce, pause):
    def nonce():
        return uuid.uuid4().hex[:12] if use_nonce else None

    print("\nWarm-up (not counted): one request without the image, one with it")
    base = timed_call(style, endpoint, key, api_name, QUESTIONS[0][3] + SUFFIX, None, nonce())
    time.sleep(pause)
    warm = timed_call(style, endpoint, key, api_name, QUESTIONS[0][3] + SUFFIX, image, nonce())
    if not warm["ok"]:
        sys.exit(f"  The image request failed: {warm['status']} {warm['error']}")
    image_tokens = None
    if base["ok"] and base.get("input_tokens") and warm.get("input_tokens"):
        image_tokens = warm["input_tokens"] - base["input_tokens"]
        print(f"  This model reads the image as {image_tokens} tokens")

    runs = []
    for rnd in range(1, rounds + 1):
        order = QUESTIONS[:]
        random.shuffle(order)                                  # new order every round
        print(f"\n  Round {rnd}/{rounds}: ", end="", flush=True)
        for qid, cat, group, question, truth in order:
            time.sleep(pause)
            r = timed_call(style, endpoint, key, api_name, question + SUFFIX, image, nonce())
            # --- clock has stopped; everything below is bookkeeping ---
            said = read_yes_no(r["text"]) if r["ok"] else "error"
            r.update(round=rnd, qid=qid, category=cat, group=group, question=question,
                     truth=truth, said=said, correct=(said == truth))
            runs.append(r)
            mark = "ok" if r["correct"] else ("ERR" if not r["ok"] else "x")
            print(f"{qid} {mark} {r.get('total', 0):.1f}s  ", end="", flush=True)
        print()
    return runs, image_tokens


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def pct(n, d):
    return 100.0 * n / d if d else None


def analyse(runs, rounds):
    qids = [q[0] for q in QUESTIONS]
    cats = list(dict.fromkeys(q[1] for q in QUESTIONS))
    group_of = {q[1]: q[2] for q in QUESTIONS}

    per_q = []
    for qid, cat, group, question, truth in QUESTIONS:
        rs = [r for r in runs if r["qid"] == qid]
        answers = [r["said"] for r in rs]
        counts = Counter(answers).most_common()
        top_n = counts[0][1] if counts else 0
        top = "/".join(a for a, n in counts if n == top_n)      # shows ties, e.g. "yes/no"
        per_q.append(dict(
            qid=qid, category=cat, group=group, question=question, truth=truth, asked=len(rs),
            accuracy=pct(sum(r["correct"] for r in rs), len(rs)),
            yes=answers.count("yes"), no=answers.count("no"), invalid=answers.count("invalid"),
            errors=answers.count("error"), usual=top, consistency=pct(top_n, len(rs)),
            first_token=med([r.get("first_token") for r in rs]), total=med([r.get("total") for r in rs]),
            total_min=min((r["total"] for r in rs if r.get("total") is not None), default=None),
            total_max=max((r["total"] for r in rs if r.get("total") is not None), default=None),
            out_tokens=med([r.get("output_tokens") for r in rs]),
            reasoning=med([r.get("reasoning_tokens") for r in rs]),
            example=next((r["text"] for r in rs if r.get("text")), "")))

    def pair_plus(cat):
        """MME accuracy+: per round, both questions of the pair right."""
        ids = [q[0] for q in QUESTIONS if q[1] == cat]
        both = 0
        for rnd in range(1, rounds + 1):
            rs = [r for r in runs if r["round"] == rnd and r["qid"] in ids]
            both += len(rs) == len(ids) and all(r["correct"] for r in rs)
        return pct(both, rounds)

    per_cat = []
    for cat in cats:
        rs = [r for r in runs if r["category"] == cat]
        acc, accp = pct(sum(r["correct"] for r in rs), len(rs)), pair_plus(cat)
        per_cat.append(dict(category=cat, group=group_of[cat], asked=len(rs), accuracy=acc,
                            accuracy_plus=accp, score=(acc or 0) + (accp or 0),
                            first_token=med([r.get("first_token") for r in rs]),
                            total=med([r.get("total") for r in rs]),
                            out_tokens=med([r.get("output_tokens") for r in rs]),
                            reasoning=med([r.get("reasoning_tokens") for r in rs])))

    ok = [r for r in runs if r["ok"]]
    answered = [r for r in runs if r["said"] in ("yes", "no")]
    summary = dict(
        requests=len(runs), failed=len(runs) - len(ok),
        perception=sum(c["score"] for c in per_cat if c["group"] == "Perception"),
        perception_max=200 * sum(1 for c in per_cat if c["group"] == "Perception"),
        cognition=sum(c["score"] for c in per_cat if c["group"] == "Cognition"),
        cognition_max=200 * sum(1 for c in per_cat if c["group"] == "Cognition"),
        accuracy=pct(sum(r["correct"] for r in runs), len(runs)),
        yes_ratio=pct(sum(r["said"] == "yes" for r in answered), len(answered)),
        invalid=sum(r["said"] == "invalid" for r in runs),
        always_same=sum(1 for q in per_q if q["consistency"] == 100),
        first_token=med([r.get("first_token") for r in ok]), total=med([r.get("total") for r in ok]),
        network=med([r["dns"] + r["tcp"] + r["tls"] for r in ok if r.get("tls") is not None]),
        rtt=min((r["tcp"] for r in ok if r.get("tcp")), default=None),
        upload=med([r.get("upload") for r in ok]),
        input_tokens=med([r.get("input_tokens") for r in ok]),
        out_tokens=med([r.get("output_tokens") for r in ok]),
        reasoning=med([r.get("reasoning_tokens") for r in ok]),
        cached=max((r.get("cached_tokens") or 0 for r in ok), default=0),
        server_timing=next((r["server_timing"] for r in ok if r["server_timing"]), ""))
    return per_q, per_cat, summary, qids


# Report
def f(x, nd=2):
    return "" if x is None else round(x, nd)


def s(x, fmt="{:.2f}"):
    return "   -" if x is None else fmt.format(x)


def print_report(label, image_name, rounds, per_q, per_cat, sm, image_tokens):
    line = "=" * 100
    print(f"\n{line}\nMME-STYLE RESULTS  {label}  |  {image_name}  |  {rounds} rounds x "
          f"{len(QUESTIONS)} questions\n{line}")
    print(f"{'Category':<24}{'Accuracy':>9}{'Acc+':>7}{'Score':>7}   "
          f"{'1st word':>9}{'Total':>8}{'Out tok':>8}{'Reason':>8}")
    for c in per_cat:
        print(f"{c['category']:<24}{s(c['accuracy'], '{:.0f}%'):>9}{s(c['accuracy_plus'], '{:.0f}%'):>7}"
              f"{s(c['score'], '{:.0f}'):>7}   {s(c['first_token'], '{:.2f}s'):>9}{s(c['total'], '{:.2f}s'):>8}"
              f"{s(c['out_tokens'], '{:.0f}'):>8}{s(c['reasoning'], '{:.0f}'):>8}")
    print(f"\n{'Question':<6}{'Truth':>6}{'Usual':>8}{'Acc':>6}{'Same':>6}{'Total':>8}   Question")
    for q in per_q:
        print(f"{q['qid']:<6}{q['truth']:>6}{q['usual']:>8}{s(q['accuracy'], '{:.0f}%'):>6}"
              f"{s(q['consistency'], '{:.0f}%'):>6}{s(q['total'], '{:.2f}s'):>8}   {q['question']}")
    print(f"\nMME score: perception {sm['perception']:.0f} / {sm['perception_max']}, "
          f"cognition {sm['cognition']:.0f} / {sm['cognition_max']}   |   overall accuracy "
          f"{s(sm['accuracy'], '{:.0f}%')}   |   says 'yes' {s(sm['yes_ratio'], '{:.0f}%')} of the time "
          f"(fair = 50%)")
    print(f"Not yes/no (counted wrong): {sm['invalid']}   |   failed requests: {sm['failed']}   |   "
          f"questions answered the same every round: {sm['always_same']} / {len(per_q)}")
    print(f"Median times: first word {s(sm['first_token'])} s, total {s(sm['total'])} s   |   network setup "
          f"{s(sm['network'], '{:.3f}')} s, round trip {s(sm['rtt'], '{:.3f}')} s, upload "
          f"{s(sm['upload'], '{:.3f}')} s")
    print(f"Tokens: image {image_tokens if image_tokens is not None else '-'}, input {s(sm['input_tokens'], '{:.0f}')}, "
          f"output {s(sm['out_tokens'], '{:.0f}')}, reasoning {s(sm['reasoning'], '{:.0f}')}")
    if sm["cached"]:
        print(f"WARNING: the provider reused cached work ({sm['cached']} tokens); times may be too fast.")


# Results file
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(v):
    return ILLEGAL.sub("", v)[:32000] if isinstance(v, str) else v


def build_sheets(session, label, api_name, image_name, rounds, runs, per_q, per_cat, sm, image_tokens):
    key = [session, label]
    summary = (["Session", "Model", "API name", "Image", "Rounds", "Requests", "Failed",
                "Perception score", "Perception max", "Cognition score", "Cognition max",
                "Accuracy %", "Says yes %", "Not yes/no", "Questions always same answer",
                "Median first word (s)", "Median total (s)", "Network setup (s)", "Round trip (s)",
                "Upload (s)", "Image tokens", "Input tokens", "Output tokens", "Reasoning tokens",
                "Cached tokens", "Server timing headers"],
               [key + [api_name, image_name, rounds, sm["requests"], sm["failed"],
                       f(sm["perception"], 1), sm["perception_max"], f(sm["cognition"], 1),
                       sm["cognition_max"], f(sm["accuracy"], 1), f(sm["yes_ratio"], 1), sm["invalid"],
                       sm["always_same"], f(sm["first_token"]), f(sm["total"]), f(sm["network"], 3),
                       f(sm["rtt"], 3), f(sm["upload"], 3), image_tokens, f(sm["input_tokens"], 0),
                       f(sm["out_tokens"], 0), f(sm["reasoning"], 0), sm["cached"], sm["server_timing"]]])
    category = (["Session", "Model", "Category", "MME group", "Questions asked", "Accuracy %",
                 "Accuracy+ %", "MME score (max 200)", "Median first word (s)", "Median total (s)",
                 "Median output tokens", "Median reasoning tokens"],
                [key + [c["category"], c["group"], c["asked"], f(c["accuracy"], 1),
                        f(c["accuracy_plus"], 1), f(c["score"], 1), f(c["first_token"]), f(c["total"]),
                        f(c["out_tokens"], 0), f(c["reasoning"], 0)] for c in per_cat])
    question = (["Session", "Model", "Q", "Category", "Question", "Correct answer", "Usual answer",
                 "Accuracy %", "Same answer % (consistency)", "Yes", "No", "Not yes/no", "Failed",
                 "Median first word (s)", "Median total (s)", "Fastest (s)", "Slowest (s)",
                 "Median output tokens", "Median reasoning tokens", "Example reply"],
                [key + [q["qid"], q["category"], q["question"], q["truth"], q["usual"],
                        f(q["accuracy"], 1), f(q["consistency"], 1), q["yes"], q["no"], q["invalid"],
                        q["errors"], f(q["first_token"]), f(q["total"]), f(q["total_min"]),
                        f(q["total_max"]), f(q["out_tokens"], 0), f(q["reasoning"], 0), q["example"]]
                 for q in per_q])
    run_rows = (["Session", "Model", "Round", "Q", "Category", "Question", "Correct answer", "Model said",
                 "Correct?", "Full reply", "DNS (s)", "TCP = round trip (s)", "TLS (s)", "Upload (s)",
                 "First word (s)", "Total (s)", "Input tokens", "Output tokens", "Reasoning tokens",
                 "Cached tokens", "Error"],
                [key + [r["round"], r["qid"], r["category"], r["question"], r["truth"], r["said"],
                        "yes" if r["correct"] else "no", r.get("text", ""), f(r.get("dns"), 3),
                        f(r.get("tcp"), 3), f(r.get("tls"), 3), f(r.get("upload"), 3),
                        f(r.get("first_token"), 3), f(r.get("total"), 3), r.get("input_tokens", ""),
                        r.get("output_tokens", ""), r.get("reasoning_tokens", ""),
                        r.get("cached_tokens", ""), "" if r["ok"] else f"{r['status']}: {r['error']}"]
                 for r in sorted(runs, key=lambda x: (x["round"], int(x["qid"][1:])))])
    return {"Summary": summary, "Per category": category, "Per question": question, "Runs": run_rows}


def save(sheets):
    sheets = {n: (h, [[clean(v) for v in row] for row in rows]) for n, (h, rows) in sheets.items()}
    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print("\n(openpyxl is not installed, so CSV files are written. For Excel: pip3 install openpyxl)")
        return save_csv(sheets)
    try:
        if RESULTS_FILE.exists():
            wb = load_workbook(RESULTS_FILE)
        else:
            wb = Workbook()
            wb.remove(wb.active)
        for name, (headers, rows) in sheets.items():
            if name not in wb.sheetnames:
                ws = wb.create_sheet(name)
                ws.append(headers)
                for c in ws[1]:
                    c.font = Font(bold=True)
                    c.fill = PatternFill("solid", fgColor="FFFF00")
                    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                for i, h in enumerate(headers, 1):
                    wide = 50 if h in ("Question", "Full reply", "Example reply") else \
                        24 if h in ("Category", "Server timing headers") else 13
                    ws.column_dimensions[ws.cell(1, i).column_letter].width = wide
                ws.freeze_panes = "A2"
            ws = wb[name]
            for row in rows:
                ws.append(row)
        wb.save(RESULTS_FILE)
        print(f"\nSaved to {RESULTS_FILE}")
    except PermissionError:
        print(f"\n!! {RESULTS_FILE.name} is open in Excel; close it next time. Writing CSV files instead.")
        save_csv(sheets)


def save_csv(sheets):
    import csv
    for name, (headers, rows) in sheets.items():
        p = HERE / f"results_mme_{name.lower().replace(' ', '_')}.csv"
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(headers)
            w.writerows(rows)
        print(f"Saved to {p}")


# Main
def main():
    ap = argparse.ArgumentParser(description="MME-style yes/no questions with timing")
    ap.add_argument("--model", help='model to test, e.g. "GPT-4o" (asks if left out)')
    ap.add_argument("--image", default=IMAGE, help="image name in the images folder, or a file path")
    ap.add_argument("--rounds", type=int, default=ROUNDS, help="rounds of all 10 questions")
    ap.add_argument("--pause", type=float, default=PAUSE, help="seconds between requests")
    ap.add_argument("--allow-cache", action="store_true",
                    help="don't add the unique request id (lets the provider reuse cached work)")
    ap.add_argument("--recheck", action="store_true", help="find the API name again")
    ap.add_argument("--reset", action="store_true", help="forget saved endpoint and API names")
    args = ap.parse_args()

    labels = [m[0] for m in MODELS]
    if args.model:
        lookup = {l.lower().replace(" ", ""): l for l in labels}
        label = lookup.get(args.model.lower().replace(" ", ""))
        if not label:
            sys.exit(f"Unknown model '{args.model}'. Use one of: {', '.join(labels)}")
    else:
        for i, l in enumerate(labels, 1):
            print(f"  {i}. {l}")
        try:
            label = labels[int(input("Which model? Type its number: ").strip()) - 1]
        except (ValueError, IndexError):
            sys.exit("Please type one of the numbers shown.")

    image_path = find_image(args.image)
    if image_path is None:
        sys.exit(f"Image '{args.image}' not found (looked in {IMAGES_DIR}).")
    image = load_image(image_path)

    if args.reset and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    cfg = load_config()
    endpoint = get_endpoint(cfg)
    key = get_key()
    found = cfg.get("found", {})
    if args.recheck or label not in found:
        found.pop(label, None)
        found.update(find_models(endpoint, key, [label]))
        cfg["found"] = {l: v for l, v in found.items() if l in labels}
        save_config(cfg)
    if label not in found:
        sys.exit(f"\n{label} did not answer. Put the exact API name first in its MODELS list, "
                 "then run again with --recheck.")

    api_name, style = found[label]["api_name"], found[label]["style"]
    rounds = max(1, args.rounds)
    print(f"\n=== {label} (API name '{api_name}') | {image_path.name} ===")
    print(f"{rounds} rounds x {len(QUESTIONS)} questions = {rounds * len(QUESTIONS)} requests "
          f"(+2 warm-up). Keep Excel closed.")
    runs, image_tokens = run_rounds(endpoint, key, api_name, style, image, rounds,
                                    not args.allow_cache, args.pause)
    per_q, per_cat, sm, _ = analyse(runs, rounds)
    print_report(label, image_path.name, rounds, per_q, per_cat, sm, image_tokens)
    session = datetime.now().strftime("%Y-%m-%d %H:%M")
    save(build_sheets(session, label, api_name, image_path.name, rounds, runs, per_q, per_cat, sm,
                      image_tokens))


if __name__ == "__main__":
    main()
