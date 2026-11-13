#!/usr/bin/env python3
# Experiment 03: four kinds of request, to estimate where a model spends its time.
# Run: python3 test_03.py --model "Llama-4 Maverick" --repeat 10
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
from datetime import datetime
from pathlib import Path

# Settings
IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "test_images"
HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "results_timing_v2.xlsx"
CONFIG_FILE = HERE / "patriot_api_config.json"        # endpoint + found API names (never the key)

IMAGE = "IMG_1"
PROMPT = "What big cat is this?"

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

REPEAT = 5                    # rounds of A+B+C+D; the report uses medians
TIMEOUT = 180
PAUSE = 1.0                   # seconds between requests
MAX_IMAGE_BYTES = 3_500_000
ANTHROPIC_VERSION = "2023-06-01"
FILLER = "The quick brown fox jumps over the lazy dog near the quiet river bank today. "

STYLES_OPENAI = ["v1", "v1-preview", "deploy-2025", "deploy-2024"]
STYLES_CLAUDE = ["anthropic"] + STYLES_OPENAI
TIMING_HEADER = re.compile(r"(time|ms\b|-ms|duration|latency|processing|elapsed)", re.I)
VARIANTS = {
    "A": "tiny text, 1 token",
    "B": "long text, 1 token",
    "C": "image, 1 token",
    "D": "image, full answer",
}


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
# Remembered per (api_name, style): whether the server accepts stream_options, and which
# name it uses for the answer-length cap.
QUIRKS = {}


def build_request(style, endpoint, key, api_name, prompt, image, max_tokens, nonce, quirks):
    tag = f"Request id {nonce}." if nonce else None      # unique start = no prompt-cache reuse
    if style == "anthropic":
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64",
                                                           "media_type": image[0], "data": image[1]}})
        payload = {"model": api_name, "max_tokens": max_tokens or 1024, "stream": True,
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
        payload["stream_options"] = {"include_usage": True}
    if max_tokens:
        payload[quirks.get("max_param", "max_tokens")] = max_tokens
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
    """Bytes we sent that the other side has NOT yet confirmed receiving (None = can't tell)."""
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
    """Returns the time (since t0) at which every uploaded byte was confirmed received."""
    if bytes_not_yet_acknowledged(sock) is None:
        return None
    deadline = time.perf_counter() + limit
    while time.perf_counter() < deadline:
        n = bytes_not_yet_acknowledged(sock)
        if not n:
            return time.perf_counter() - t0
        time.sleep(0.0005)
    return None


# One timed request
def measure(style, endpoint, key, api_name, prompt, image, max_tokens, nonce):
    quirks = QUIRKS.setdefault((api_name, style), {})
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image,
                                          max_tokens, nonce, quirks)
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    host, https = u.hostname, u.scheme == "https"
    port = u.port or (443 if https else 80)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    r = dict(ok=False, status=None, error="", text="", bytes_sent=len(body), streamed=True,
             server_timing="")
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
                     "Content-Length": str(len(body)), "User-Agent": "patriot-timing/2.0"}.items():
            conn.putheader(k, v)
        r["send_start"] = time.perf_counter() - t0
        conn.endheaders(body)
        r["send_returned"] = time.perf_counter() - t0              # handed to the OS
        r["upload_acked"] = wait_until_acknowledged(sock, t0)      # actually received

        resp = conn.getresponse()
        r["headers_at"] = time.perf_counter() - t0
        r["status"] = resp.status
        r["server_timing"] = "; ".join(f"{k}={v}" for k, v in resp.getheaders()
                                       if TIMING_HEADER.search(k) and k.lower() != "date")
        if resp.status >= 400:
            r["error"] = short_error(resp.read().decode("utf-8", "replace"))
            return r

        if "event-stream" not in (resp.getheader("Content-Type") or ""):
            raw = resp.read().decode("utf-8", "replace")
            r["end"] = r["first_token"] = r["last_token"] = time.perf_counter() - t0
            r["streamed"] = False
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
                if kind == "thinking":
                    r.setdefault("first_thinking", now)
                else:
                    r.setdefault("first_answer", now)
                    answer.append(piece)
                r["last_token"] = now
            if style == "anthropic" and ev.get("type") == "message_stop":
                break
        r["end"] = time.perf_counter() - t0                       # clock stops
        r.update(acc, ok=True, text="".join(answer))
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    finally:
        if conn is not None:
            conn.close()


def timed_call(style, endpoint, key, api_name, prompt, image=None, max_tokens=None, nonce=None):
    """measure(), adapting once if the server rejects stream_options or the max_tokens name."""
    quirks = QUIRKS.setdefault((api_name, style), {})
    for _ in range(3):
        r = measure(style, endpoint, key, api_name, prompt, image, max_tokens, nonce)
        err = (r["error"] or "").lower()
        if r["ok"] or r["status"] != 400:
            return r
        if "stream_options" in err and quirks.get("usage", True):
            quirks["usage"] = False
        elif max_tokens and "max_tokens" in err and quirks.get("max_param") != "max_completion_tokens":
            quirks["max_param"] = "max_completion_tokens"
        else:
            return r
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


# The experiment
def server_time(r, rtt):
    """Time the request spent at the provider: done - upload received - 1 network trip."""
    up = r.get("upload_acked") if r.get("upload_acked") is not None else r.get("send_returned")
    if r.get("end") is None or up is None:
        return None
    return max(0.0, r["end"] - up - rtt)


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def spread(vals):
    vals = [v for v in vals if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def run_experiment(endpoint, key, api_name, style, image, prompt, repeat, use_nonce, pause):
    def call(v, filler=""):
        nonce = uuid.uuid4().hex[:12] if use_nonce else None
        if v == "A":
            return timed_call(style, endpoint, key, api_name, "Reply with one word: OK", None, 1, nonce)
        if v == "B":
            return timed_call(style, endpoint, key, api_name,
                              filler + "\nReply with one word: OK", None, 1, nonce)
        if v == "C":
            return timed_call(style, endpoint, key, api_name, prompt, image, 1, nonce)
        return timed_call(style, endpoint, key, api_name, prompt, image, None, nonce)

    print("\nWarm-up (not counted): wakes the connection path and measures the image's token size")
    call("A")
    time.sleep(pause)
    cal = call("C")
    if not cal["ok"]:
        sys.exit(f"  The 1-token image request failed: {cal['status']} {cal['error']}\n"
                 "  (Reasoning models sometimes refuse a 1-token cap; try another model.)")
    img_tokens = cal.get("input_tokens") or 1500
    filler = FILLER * max(1, round(img_tokens / 15))          # ~15 tokens per filler sentence
    print(f"  image request = {img_tokens} input tokens; long-text test sized to match")

    runs = []
    for rep in range(1, repeat + 1):
        order = list(VARIANTS)
        random.shuffle(order)                                  # no test always goes first
        print(f"\n  Round {rep}/{repeat}: ", end="", flush=True)
        for v in order:
            time.sleep(pause)
            r = call(v, filler)
            r.update(variant=v, round=rep)
            runs.append(r)
            print(f"{v} {r['end']:.2f}s  " if r["ok"] else f"{v} ERROR {r['status']}  ",
                  end="", flush=True)
        print()
    return runs


def analyse(runs):
    ok = [r for r in runs if r["ok"]]
    rtt = min((r["tcp"] for r in ok if r.get("tcp")), default=0.0)   # best round trip seen
    by = {v: [r for r in ok if r["variant"] == v] for v in VARIANTS}
    st = {v: [server_time(r, rtt) for r in by[v]] for v in VARIANTS}
    tok = {v: med([r.get("input_tokens") for r in by[v]]) for v in VARIANTS}
    out_d = med([r.get("output_tokens") for r in by["D"]])
    m = {v: med(st[v]) for v in VARIANTS}

    rows = []   # (stage, what it covers, value, unit, type, range/notes)

    def add(stage, covers, value, unit, kind, note=""):
        rows.append((stage, covers, value, unit, kind, note))

    def rng(vals, unit="s"):
        lo, hi = spread(vals)
        return "" if lo is None else f"{lo:.3f}-{hi:.3f} {unit} across runs"

    allr = ok
    add("Connection setup", "find server + open connection + encryption (DNS+TCP+TLS)",
        med([r["dns"] + r["tcp"] + r["tls"] for r in allr]), "s", "MEASURED",
        rng([r["dns"] + r["tcp"] + r["tls"] for r in allr]))
    add("Network round trip", "one message there and back (best TCP connect)", rtt, "s", "MEASURED")
    ups = [r["upload_acked"] - r["send_start"] for r in by["C"] + by["D"]
           if r.get("upload_acked") is not None]
    if ups:
        add("Upload of prompt + image", f"until the server confirmed receiving all "
            f"{med([r['bytes_sent'] for r in by['C']]) / 1024:.0f} KB", med(ups), "s", "MEASURED", rng(ups))
    else:
        add("Upload of prompt + image", "this computer can't report it; it is inside the server times",
            None, "s", "-")

    add("1. Gathering (fixed overhead)",
        "accept + check the request, queue, start the model; the cost of ANY request (test A)",
        m["A"], "s", "DERIVED", rng(st["A"]))
    per_k = None
    if None not in (m["A"], m["B"], tok["A"], tok["B"]) and tok["B"] > tok["A"]:
        per_k = (m["B"] - m["A"]) / (tok["B"] - tok["A"]) * 1000
        add("2. Reading text", f"per 1,000 tokens of text (B - A, {tok['B'] - tok['A']:.0f} tokens)",
            per_k, "s", "DERIVED", rng(st["B"]))
    img = None
    if None not in (m["A"], m["C"]):
        img = m["C"] - m["A"]
        ctok = f"{tok['C']:.0f}" if tok["C"] else "?"
        add("3. Understanding the image + prompt",
            f"preprocessing + vision encoder + reading {ctok} tokens (C - A)",
            img, "s", "DERIVED", rng(st["C"]))
    if img is not None and per_k is not None and tok["C"] and tok["A"]:
        same_as_text = per_k * (tok["C"] - tok["A"]) / 1000
        add("   3a. ...of which: reading those tokens", "what the same number of TEXT tokens costs",
            same_as_text, "s", "DERIVED")
        add("   3b. ...of which: image-only work",
            "preprocessing + vision encoder (3 minus 3a); rough, can be noisy", img - same_as_text,
            "s", "DERIVED")
    if None not in (out_d,) and by["C"] and by["D"] and out_d > 1:
        per_tok = (med([r["end"] for r in by["D"]]) - med([r["end"] for r in by["C"]])) / (out_d - 1)
        add("4. Writing the answer", f"per token (D - C over {out_d - 1:.0f} extra tokens)",
            per_tok * 1000, "ms", "DERIVED", f"= {1 / per_tok:.0f} tokens/s" if per_tok > 0 else "")
        add("   writing this answer", f"{out_d:.0f} tokens", per_tok * (out_d - 1), "s", "DERIVED")
    add("5. Giving back the result", "each piece travels back while the next is written; "
        "only the last trip adds time (half a round trip)", rtt / 2, "s", "DERIVED")
    add("Start to finish (normal answer)", "test D: request start -> last token",
        med([r["end"] for r in by["D"]]), "s", "MEASURED", rng([r["end"] for r in by["D"]]))
    add("First word appeared (normal answer)", "test D",
        med([r.get("first_token") for r in by["D"]]), "s", "MEASURED")
    add("Image size in tokens", "C input tokens - A input tokens, reported by the server",
        (tok["C"] - tok["A"]) if tok["C"] and tok["A"] else None, "tokens", "MEASURED")

    heads = sorted({r["server_timing"] for r in ok if r["server_timing"]})
    if heads:
        add("Server-reported timing headers", "the provider's own numbers, see Runs sheet",
            None, "", "MEASURED", heads[0][:200])
    cached = [r.get("cached_tokens") for r in ok if r.get("cached_tokens")]
    if cached:
        add("WARNING: prompt cache used", "some reading was skipped; results too fast",
            max(cached), "tokens", "-")
    add("Queue vs actual work", "waiting in line vs computing: not visible through an API",
        None, "", "-")
    add("Preprocessing vs vision encoder", "not separable through an API (both are in 3b)",
        None, "", "-")

    noise = None
    if m["A"] is not None and img is not None:
        lo, hi = spread(st["A"])
        if hi - lo > abs(img):
            noise = ("The variation between runs is larger than the image cost itself. "
                     "Run again with --repeat 10 or more.")
    return rows, noise


# Output
def print_report(rows, noise, label, image_name, prompt, repeat):
    print(f"\n{'=' * 96}\nTIMING BREAKDOWN  {label}  |  {image_name}  |  \"{prompt}\"  |  "
          f"median of {repeat} rounds\n{'=' * 96}")
    for stage, covers, value, unit, kind, note in rows:
        v = "      -" if value is None else (f"{value:7.0f}" if unit == "tokens" else
                                              f"{value:7.1f}" if unit == "ms" else f"{value:7.3f}")
        print(f"{stage:<38}{v} {unit:<7}{kind:<9} {covers}")
        if note:
            print(f"{'':<54}{note}")
    if noise:
        print(f"\nNOTE: {noise}")


ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(v):
    return ILLEGAL.sub("", v)[:32000] if isinstance(v, str) else v


RUN_HEADERS = ["Session", "Model", "Round", "Test", "Test means", "Request KB", "DNS (s)",
               "TCP = round trip (s)", "TLS (s)", "Upload confirmed (s)", "Response started (s)",
               "First token (s)", "Done (s)", "Server time (s)", "Input tokens", "Cached tokens",
               "Output tokens", "Reasoning tokens", "Server timing headers", "Error", "AI response"]


def save(session, label, runs, rows, image_name, prompt):
    rtt = min((r["tcp"] for r in runs if r["ok"] and r.get("tcp")), default=0.0)

    def f(x, nd=3):
        return "" if x is None else round(x, nd)
    run_rows = [[session, label, r["round"], r["variant"], VARIANTS[r["variant"]],
                 round(r["bytes_sent"] / 1024), f(r.get("dns")), f(r.get("tcp")), f(r.get("tls")),
                 f(r["upload_acked"] - r["send_start"]) if r.get("upload_acked") is not None else "",
                 f(r.get("headers_at")), f(r.get("first_token")), f(r.get("end")),
                 f(server_time(r, rtt)) if r["ok"] else "", r.get("input_tokens", ""),
                 r.get("cached_tokens", ""), r.get("output_tokens", ""), r.get("reasoning_tokens", ""),
                 r["server_timing"], "" if r["ok"] else f"{r['status']}: {r['error']}", r.get("text", "")]
                for r in runs]
    bd_headers = ["Session", "Model", "Image", "Prompt", "Stage", "What it covers", "Value", "Unit",
                  "Type", "Spread / note"]
    bd_rows = [[session, label, image_name, prompt, s, c, f(v, 4), u, k, n] for s, c, v, u, k, n in rows]
    run_rows = [[clean(x) for x in row] for row in run_rows]
    bd_rows = [[clean(x) for x in row] for row in bd_rows]
    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print("\n(openpyxl is not installed, so CSV files are written instead. "
              "For Excel: pip3 install openpyxl)")
        return save_csv(run_rows, bd_rows, RUN_HEADERS, bd_headers)

    def sheet(wb, name, headers, widths):
        if name in wb.sheetnames:
            return wb[name]
        ws = wb.create_sheet(name)
        ws.append(headers)
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="FFFF00")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = w
        ws.freeze_panes = "A2"
        return ws
    try:
        if RESULTS_FILE.exists():
            wb = load_workbook(RESULTS_FILE)
        else:
            wb = Workbook()
            wb.remove(wb.active)
        ws = sheet(wb, "Breakdown", bd_headers, [17, 18, 12, 30, 36, 60, 10, 8, 11, 40])
        for row in bd_rows:
            ws.append(row)
        ws = sheet(wb, "Runs", RUN_HEADERS, [17, 18, 7, 6, 20] + [11] * 13 + [40, 30, 60])
        for row in run_rows:
            ws.append(row)
        wb.save(RESULTS_FILE)
        print(f"\nSaved to {RESULTS_FILE}")
    except PermissionError:
        print(f"\n!! {RESULTS_FILE.name} is open in Excel; close it next time. Writing CSV files instead.")
        save_csv(run_rows, bd_rows, RUN_HEADERS, bd_headers)


def save_csv(run_rows, bd_rows, run_headers, bd_headers):
    import csv
    for name, headers, rows in (("results_timing_v2_breakdown.csv", bd_headers, bd_rows),
                                ("results_timing_v2_runs.csv", run_headers, run_rows)):
        p = HERE / name
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(headers)
            w.writerows(rows)
        print(f"Saved to {p}")


# Main
def main():
    ap = argparse.ArgumentParser(description="Patriot AI timing breakdown v2 (1 image, 1 prompt)")
    ap.add_argument("--model", help='model to test, e.g. "GPT-4o" (asks if left out)')
    ap.add_argument("--image", default=IMAGE, help="image name in the images folder, or a file path")
    ap.add_argument("--prompt", default=PROMPT, help="the question to ask")
    ap.add_argument("--repeat", type=int, default=REPEAT, help="rounds of A+B+C+D (median reported)")
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
    repeat = max(1, args.repeat)
    print(f"\n=== {label} (API name '{api_name}') | {image_path.name} | \"{args.prompt}\" ===")
    print(f"{repeat} rounds x 4 tests = {repeat * 4} requests (+2 warm-up). Keep Excel closed.")
    runs = run_experiment(endpoint, key, api_name, style, image, args.prompt, repeat,
                          not args.allow_cache, args.pause)
    rows, noise = analyse(runs)
    print_report(rows, noise, label, image_path.name, args.prompt, repeat)
    save(datetime.now().strftime("%Y-%m-%d %H:%M"), label, runs, rows, image_path.name, args.prompt)


if __name__ == "__main__":
    main()
