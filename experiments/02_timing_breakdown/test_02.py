#!/usr/bin/env python3
# Experiment 02: one photo and one question, with the response time split into its parts.
# Run: python3 test_02.py --model "Llama-4 Maverick" --repeat 3
import argparse
import base64
import getpass
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

# Settings
IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "test_images"
HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "results_timing.xlsx"
FALLBACK_CSV = HERE / "results_timing_fallback.csv"
CONFIG_FILE = HERE / "patriot_api_config.json"        # endpoint + found API names (never the key)

# The one image and prompt to test (change these later). IMAGE is a name in IMAGES_DIR
# (without extension, e.g. "IMG_1") or a full path to a file.
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

REPEAT = 3                    # runs of each variant; the summary uses the median
TIMEOUT = 180                 # seconds before one answer is given up
PAUSE = 2.0                   # seconds between requests
MAX_IMAGE_BYTES = 3_500_000   # bigger photos are shrunk with macOS 'sips'
ANTHROPIC_VERSION = "2023-06-01"

STYLES_OPENAI = ["v1", "v1-preview", "deploy-2025", "deploy-2024"]
STYLES_CLAUDE = ["anthropic"] + STYLES_OPENAI

RUN_HEADERS = [
    "Session", "Model", "API name", "Variant", "Repeat",
    "Request size (KB)", "DNS (s)", "TCP connect (s)", "TLS (s)", "Upload (s)", "Upload speed (Mbps)",
    "Headers received (s)", "First token (s)", "Server before 1st token, est. (s)",
    "Visible thinking (s)", "Answer streaming (s)", "Stream close (s)", "Total (s)",
    "Input tokens", "Output tokens", "Reasoning tokens", "Answer tokens/s", "Chunks",
    "Median gap between chunks (ms)", "Notes", "Image", "Prompt", "AI response",
]
SUMMARY_HEADERS = ["Session", "Model", "Variant", "Runs OK", "Network setup (s)", "Upload (s)",
                   "First token (s)", "Server before 1st token (s)", "Visible thinking (s)",
                   "Answer streaming (s)", "Total (s)", "Input tokens", "Output tokens",
                   "Answer tokens/s"]


# Config
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


# Image
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
    """Returns (mime, base64). Shrinks big photos / converts HEIC using macOS 'sips'."""
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
def build_request(style, endpoint, key, api_name, prompt, image, include_usage=True):
    if style == "anthropic":
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64",
                                                           "media_type": image[0], "data": image[1]}})
        payload = {"model": api_name, "max_tokens": 1024, "stream": True,
                   "messages": [{"role": "user", "content": content}]}
        return (f"{endpoint}/anthropic/v1/messages", payload,
                {"api-key": key, "x-api-key": key, "anthropic-version": ANTHROPIC_VERSION})

    if image:
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": f"data:{image[0]};base64,{image[1]}"}}]
    else:
        content = prompt
    payload = {"messages": [{"role": "user", "content": content}], "stream": True}
    if include_usage:
        payload["stream_options"] = {"include_usage": True}   # token counts at the end of the stream
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
                return msg[:200]
    except Exception:
        pass
    return (body or "").strip()[:200].replace("\n", " ")


def parse_event(style, ev, acc):
    """Reads one streamed event. Returns ('answer'|'thinking'|None, text) and fills acc with usage."""
    if not isinstance(ev, dict):
        return None, ""
    if style == "anthropic":
        t = ev.get("type")
        if t == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            acc["input_tokens"] = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
                + (u.get("cache_creation_input_tokens") or 0)
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


# The timed call
def measure(style, endpoint, key, api_name, prompt, image=None, include_usage=True):
    """One request with every phase timed. All times are seconds from the very start."""
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image, include_usage)
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    host, https = u.hostname, u.scheme == "https"
    port = u.port or (443 if https else 80)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    r = dict(ok=False, status=None, error="", text="", thinking_text="", bytes_sent=len(body),
             streamed=True, chunk_times=[])
    acc = {}
    conn = None
    try:
        t0 = time.perf_counter()                                   # clock starts
        addr = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4]
        r["dns"] = time.perf_counter() - t0

        t = time.perf_counter()
        sock = socket.create_connection(addr[:2], timeout=TIMEOUT)
        r["tcp"] = time.perf_counter() - t

        t = time.perf_counter()
        if https:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        r["tls"] = time.perf_counter() - t

        conn = (http.client.HTTPSConnection if https else http.client.HTTPConnection)(
            host, port, timeout=TIMEOUT)
        conn.sock = sock                                           # reuse our timed connection
        conn.putrequest("POST", path, skip_accept_encoding=True)
        for k, v in {**headers, "Content-Type": "application/json", "Accept": "text/event-stream",
                     "Content-Length": str(len(body)), "User-Agent": "patriot-timing/1.0"}.items():
            conn.putheader(k, v)
        t = time.perf_counter()
        conn.endheaders(body)                                      # sends prompt + image
        r["upload_end"] = time.perf_counter() - t0
        r["upload"] = time.perf_counter() - t

        resp = conn.getresponse()
        r["headers"] = time.perf_counter() - t0
        r["status"] = resp.status
        if resp.status >= 400:
            r["error"] = short_error(resp.read().decode("utf-8", "replace"))
            return r

        if "event-stream" not in (resp.getheader("Content-Type") or ""):
            raw = resp.read().decode("utf-8", "replace")
            r["end"] = r["last_token"] = time.perf_counter() - t0
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
            r["first_answer"] = r["end"]
            r.update(acc, ok=True)
            return r

        answer, thinking = [], []
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
                if kind == "thinking":
                    r.setdefault("first_thinking", now)
                    thinking.append(piece)
                else:
                    r.setdefault("first_answer", now)
                    answer.append(piece)
                    r["chunk_times"].append(now)
                r["last_token"] = now
            if style == "anthropic" and ev.get("type") == "message_stop":
                break
        r["end"] = time.perf_counter() - t0                       # clock stops
        r.update(acc, ok=True, text="".join(answer), thinking_text="".join(thinking))
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    finally:
        if conn is not None:
            conn.close()


def timed_call(style, endpoint, key, api_name, prompt, image=None):
    """measure(), retrying without stream_options if the server rejects that field."""
    r = measure(style, endpoint, key, api_name, prompt, image, include_usage=True)
    if not r["ok"] and r["status"] == 400 and "stream_options" in r["error"]:
        r = measure(style, endpoint, key, api_name, prompt, image, include_usage=False)
    return r


# Derived numbers
def derive(r):
    """Adds the phase durations people actually read."""
    d = {}
    d["setup"] = r.get("dns", 0) + r.get("tcp", 0) + r.get("tls", 0)
    rtt = r.get("tcp")                             # a TCP handshake takes one round trip
    first = min([x for x in (r.get("first_thinking"), r.get("first_answer")) if x is not None],
                default=None)
    d["first_token"] = first
    if first is not None and r.get("upload_end") is not None:
        d["server_before_first"] = max(0.0, first - r["upload_end"] - (rtt or 0))
    if r.get("first_thinking") is not None and r.get("first_answer") is not None:
        d["thinking"] = r["first_answer"] - r["first_thinking"]
    if r.get("first_answer") is not None and r.get("last_token") is not None:
        d["streaming"] = r["last_token"] - r["first_answer"]
    if r.get("last_token") is not None and r.get("end") is not None:
        d["close"] = r["end"] - r["last_token"]
    d["total"] = r.get("end")
    if r.get("upload"):
        d["mbps"] = r["bytes_sent"] * 8 / r["upload"] / 1e6

    out, reason = r.get("output_tokens"), r.get("reasoning_tokens") or 0
    answer_tokens = (out - reason) if out is not None else None
    if answer_tokens is None and r.get("text"):
        answer_tokens = len(r["text"]) / 4          # rough estimate when the server gives no counts
        d["tokens_estimated"] = True
    if answer_tokens and d.get("streaming", 0) > 0:
        d["tps"] = max(answer_tokens - 1, 0) / d["streaming"]
    ct = r.get("chunk_times") or []
    if len(ct) > 1:
        d["gap_ms"] = statistics.median(b - a for a, b in zip(ct, ct[1:])) * 1000
    d["chunks"] = len(ct)

    notes = []
    if not r.get("streamed", True):
        notes.append("server did not stream; only total time is meaningful")
    if reason and r.get("first_thinking") is None:
        notes.append(f"model reasoned privately ({reason} tokens); that time is inside "
                     "'server before 1st token'")
    if d.get("tokens_estimated"):
        notes.append("token counts not returned; tokens/s estimated from text length")
    d["notes"] = "; ".join(notes)
    return d


def fmt(x, nd=2):
    return "" if x is None else round(x, nd)


def print_timeline(r, d):
    def line(label, val, extra=""):
        v = "     -" if val is None else f"{val:6.2f}s"
        print(f"    {label:<30}{v}  {extra}")
    print(f"    {'Network setup':<30}{d['setup']:6.2f}s  "
          f"(DNS {r.get('dns', 0):.3f} + TCP {r.get('tcp', 0):.3f} + TLS {r.get('tls', 0):.3f})")
    line("Upload prompt + image", r.get("upload"),
         f"({r['bytes_sent'] / 1024:.0f} KB" + (f" at {d['mbps']:.1f} Mbps)" if d.get("mbps") else ")"))
    line("Server before 1st token (est.)", d.get("server_before_first"),
         "(queue + reading image & prompt)")
    line("Visible thinking", d.get("thinking"))
    tok = r.get("output_tokens")
    line("Answer streaming", d.get("streaming"),
         (f"({tok} tokens" if tok is not None else "(") +
         (f", {d['tps']:.0f} tokens/s)" if d.get("tps") else ")"))
    line("Stream close", d.get("close"))
    line("TOTAL", d.get("total"), f"(first token at {d['first_token']:.2f}s)"
         if d.get("first_token") is not None else "")
    if d["notes"]:
        print(f"    note: {d['notes']}")


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


# Results file
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(v):
    return ILLEGAL.sub("", v)[:32000] if isinstance(v, str) else v


def run_row(session, label, api_name, variant, rep, r, d, image_name, prompt):
    note = d["notes"] if r["ok"] else f"ERROR {r['status']}: {r['error']}"
    return [session, label, api_name, variant, rep, fmt(r["bytes_sent"] / 1024, 0),
            fmt(r.get("dns"), 3), fmt(r.get("tcp"), 3), fmt(r.get("tls"), 3), fmt(r.get("upload"), 3),
            fmt(d.get("mbps"), 1), fmt(r.get("headers")), fmt(d.get("first_token")),
            fmt(d.get("server_before_first")), fmt(d.get("thinking")), fmt(d.get("streaming")),
            fmt(d.get("close"), 3), fmt(d.get("total")), r.get("input_tokens", ""),
            r.get("output_tokens", ""), r.get("reasoning_tokens", ""), fmt(d.get("tps"), 1),
            d.get("chunks", ""), fmt(d.get("gap_ms"), 0), note, image_name, prompt, r.get("text", "")]


def save(run_rows, summary_rows):
    run_rows = [[clean(v) for v in row] for row in run_rows]
    summary_rows = [[clean(v) for v in row] for row in summary_rows]
    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        return save_csv(run_rows)

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
        ws = sheet(wb, "Runs", RUN_HEADERS, [17, 18, 24, 12, 7] + [11] * 19 + [40, 14, 36, 70])
        for row in run_rows:
            ws.append(row)
        ws = sheet(wb, "Summary", SUMMARY_HEADERS, [17, 18, 26, 8] + [12] * 10)
        for row in summary_rows:
            ws.append(row)
        wb.save(RESULTS_FILE)
        print(f"\nSaved to {RESULTS_FILE}")
    except PermissionError:
        print(f"\n!! {RESULTS_FILE.name} is open in Excel, so the runs went to {FALLBACK_CSV.name}")
        save_csv(run_rows)


def save_csv(rows):
    import csv
    new = not FALLBACK_CSV.exists()
    with open(FALLBACK_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RUN_HEADERS)
        w.writerows(rows)
    print(f"\nSaved to {FALLBACK_CSV}")


# Benchmark
def median_of(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    return statistics.median(vals) if vals else None


def run(endpoint, key, found, label, image_path, prompt, repeat, baseline, session):
    api_name, style = found[label]["api_name"], found[label]["style"]
    image = load_image(image_path)
    variants = [("with image", image)] + ([("text only", None)] if baseline else [])
    print(f"\n=== {label}  (API name '{api_name}') ===")
    print(f"Image: {image_path.name}   Prompt: {prompt}")

    rows, results = [], {v: [] for v, _ in variants}
    for rep in range(1, repeat + 1):
        for variant, img in variants:                 # alternate so both see the same conditions
            print(f"\n  [{variant}, run {rep}/{repeat}]")
            r = timed_call(style, endpoint, key, api_name, prompt, img)
            d = derive(r) if r["ok"] else {"notes": "", "setup": 0}
            if r["ok"]:
                print_timeline(r, d)
                if variant == "with image":
                    print(f"    answer: {r['text'][:100].replace(chr(10), ' ')}")
                results[variant].append({**r, **d})
            else:
                print(f"    ERROR {r['status']}: {r['error']}")
            rows.append(run_row(session, label, api_name, variant, rep, r, d, image_path.name, prompt))
            time.sleep(PAUSE)

    summary = []
    print(f"\n  MEDIANS over {repeat} run(s)")
    for variant, items in results.items():
        if not items:
            continue
        m = {k: median_of(items, k) for k in ("setup", "upload", "first_token", "server_before_first",
                                              "thinking", "streaming", "total", "input_tokens",
                                              "output_tokens", "tps")}
        print(f"    {variant:<11} first token {m['first_token'] or 0:.2f}s | server before 1st "
              f"{m['server_before_first'] or 0:.2f}s | streaming {m['streaming'] or 0:.2f}s | "
              f"total {m['total'] or 0:.2f}s")
        summary.append([session, label, variant, len(items)] +
                       [fmt(m[k]) for k in ("setup", "upload", "first_token", "server_before_first",
                                            "thinking", "streaming", "total")] +
                       [fmt(m["input_tokens"], 0), fmt(m["output_tokens"], 0), fmt(m["tps"], 1)])
        results[variant] = m

    a, b = results.get("with image"), results.get("text only")
    if isinstance(a, dict) and isinstance(b, dict) and a.get("server_before_first") is not None \
            and b.get("server_before_first") is not None:
        cost = a["server_before_first"] - b["server_before_first"]
        toks = (a["input_tokens"] - b["input_tokens"]) if a.get("input_tokens") and b.get("input_tokens") \
            else None
        print(f"    IMAGE COST  ~{cost:.2f}s of server time" +
              (f", ~{toks:.0f} input tokens" if toks is not None else ""))
        summary.append([session, label, "image cost (difference)", "", "", "", "", fmt(cost),
                        "", "", "", fmt(toks, 0), "", ""])
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Patriot AI timing breakdown (1 image, 1 prompt)")
    ap.add_argument("--model", help='model to test, e.g. "GPT-4o" (asks if left out)')
    ap.add_argument("--image", default=IMAGE, help="image name in the images folder, or a file path")
    ap.add_argument("--prompt", default=PROMPT, help="the question to ask")
    ap.add_argument("--repeat", type=int, default=REPEAT, help="runs per variant (median is reported)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the text-only comparison")
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
        sys.exit(f"Image '{args.image}' not found (looked in {IMAGES_DIR}).\n"
                 "If macOS asked whether Terminal may access the Desktop, click Allow and run again.")

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
        sys.exit(f"\n{label} did not answer. Put the exact API name from Chrome's Network tab first "
                 "in its list in MODELS, then run again with --recheck.")

    session = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows, summary = run(endpoint, key, found, label, image_path, args.prompt,
                        max(1, args.repeat), not args.no_baseline, session)
    save(rows, summary)


if __name__ == "__main__":
    main()
