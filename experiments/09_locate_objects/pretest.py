#!/usr/bin/env python3
# Pretest for experiment 09: asks each AI model to locate things in the jaguar photo, with no format rules,
# to see how each one answers on its own. Run: python3 pretest.py    (results: pretest_results.xlsx)
import argparse
import base64
import getpass
import http.client
import json
import os
import random
import ssl
import sys
import time
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent.parent / "test_images" / "IMG_1.jpg"
CONFIG_FILE = HERE / "patriot_api_config.json"       # endpoint + model API names (never the key)
PAUSE = 1.0                                          # seconds between requests
TIMEOUT = 180

# Sent exactly as written: no instructions about the format of the answer.
REQUESTS = [
    ("P1", "Draw a red circle around the eye of the jaguar on the right."),
    ("P2", "Crop this image to show only the eye of the jaguar on the right."),
    ("P3", "Where is the jaguar on the left?"),
    ("P4", "Where is the eye of the jaguar on the right?"),
    ("P5", "Where is the jaguar's collar?"),
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


def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def get_endpoint(cfg):
    if not cfg.get("endpoint"):
        cfg["endpoint"] = input("Endpoint URL (from the Patriot AI API Projects page): ").strip().rstrip("/")
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    return cfg["endpoint"]


def get_key():
    key = os.environ.get("PATRIOT_KEY", "").strip() or \
        getpass.getpass("Primary key (hidden: paste it and press Enter): ").strip()
    return key or sys.exit("No key entered.")


def build_request(style, endpoint, key, api_name, prompt, image, usage=True):
    first = f"Request id {uuid.uuid4().hex[:12]}."              # unique start: no cached shortcut
    if style == "anthropic":
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64",
                                                           "media_type": "image/jpeg", "data": image}})
        payload = {"model": api_name, "max_tokens": 4096, "stream": True, "system": first,
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


def ask(style, endpoint, key, api_name, prompt, image=None, usage=True):
    """One request. The connection is opened first; the clock covers sending -> last word."""
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image, usage)
    body = json.dumps(payload).encode()
    u = urllib.parse.urlsplit(url)
    r = dict(ok=False, status=None, error="", text="", first=None, total=None, out_tokens=None)
    conn = None
    try:
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=TIMEOUT,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=TIMEOUT)
        conn.connect()
        conn.putrequest("POST", u.path + (f"?{u.query}" if u.query else ""), skip_accept_encoding=True)
        for k, v in {**headers, "Content-Type": "application/json", "Accept": "text/event-stream",
                     "Content-Length": str(len(body))}.items():
            conn.putheader(k, v)
        t0 = time.perf_counter()                                     # clock starts
        conn.endheaders(body)
        resp = conn.getresponse()
        r["status"] = resp.status
        if resp.status >= 400:
            r["error"] = resp.read().decode("utf-8", "replace")[:300]
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
                r["error"] = data[:300]
                return r
            piece = ""
            if ev.get("type") == "content_block_delta":                       # Claude
                piece = (ev.get("delta") or {}).get("text") or ""
            elif ev.get("type") == "message_delta":
                r["out_tokens"] = (ev.get("usage") or {}).get("output_tokens")
            elif ev.get("choices"):                                            # GPT and Llama
                piece = (ev["choices"][0].get("delta") or {}).get("content") or ""
            if isinstance(ev.get("usage"), dict) and "completion_tokens" in ev["usage"]:
                r["out_tokens"] = ev["usage"]["completion_tokens"]
            if piece:
                r["first"] = r["first"] or time.perf_counter() - t0          # first word
                r["text"] += piece
            if ev.get("type") == "message_stop":
                break
        r["total"] = time.perf_counter() - t0                                  # last word
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
    if not r["ok"] and r["status"] == 400 and "stream_options" in r["error"]:
        model["usage"] = False                                       # this server rejects token counts
        r = ask(model["style"], endpoint, key, model["api_name"], prompt, image, False)
    return r


def find_models(endpoint, key):
    print("Finding the models' API names...")
    found = {}
    for label, company, names in MODELS:
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


def save_excel(path, rows, image_name):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "Replies"
    ws.append([f"Pretest | image {image_name} | each request sent with no format rules | times in seconds "
               f"from sending to the first / last word"])
    ws["A1"].font = Font(bold=True, size=12)
    ws.append(["Request", "Asked", "Model", "Reply", "First word (s)", "Total (s)", "Output tokens"])
    for c in ws[2]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="FFF2CC")
    order = {label: i for i, (label, _, _) in enumerate(MODELS)}
    for x in sorted(rows, key=lambda x: (x["id"], order[x["model"]])):
        ws.append([x["id"], x["prompt"], x["model"], x["reply"],
                   x["first"] and round(x["first"], 2), x["total"] and round(x["total"], 2), x["out_tokens"]])
        for c in ws[ws.max_row]:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for col, width in zip("ABCDEFG", [8, 40, 18, 90, 10, 10, 10]):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A3"
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="Pretest: how do the AI models answer location requests?")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Please install openpyxl:  python3 -m pip install openpyxl")
    if not IMAGE.is_file():
        sys.exit(f"Image not found: {IMAGE}")
    image = base64.b64encode(IMAGE.read_bytes()).decode()

    cfg = load_config()
    endpoint, key = get_endpoint(cfg), get_key()
    if len(cfg.get("found", {})) < len(MODELS):
        cfg["found"] = {**cfg.get("found", {}), **find_models(endpoint, key)}
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    models = {label: dict(cfg["found"][label]) for label, _, _ in MODELS if label in cfg["found"]}

    n = args.rounds * len(REQUESTS) * len(models)
    print(f"{n} requests ({len(REQUESTS)} requests x {len(models)} models x {args.rounds} round)")
    rows = []
    for rnd in range(args.rounds):
        for rid, prompt in random.sample(REQUESTS, len(REQUESTS)):
            for label in random.sample(list(models), len(models)):
                time.sleep(PAUSE)
                r = call(models[label], endpoint, key, prompt, image)
                rows.append(dict(id=rid, prompt=prompt, model=label, reply=r["text"] or f"FAILED: {r['error']}",
                                 first=r["first"], total=r["total"], out_tokens=r["out_tokens"]))
                print(f"  {rid} {label:<18} {'%.1f s' % r['total'] if r['total'] else 'FAILED'}")

    path = HERE / "pretest_results.xlsx"
    if path.exists():
        path = HERE / f"pretest_results_{datetime.now():%Y-%m-%d_%H%M}.xlsx"
    save_excel(path, rows, IMAGE.name)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
