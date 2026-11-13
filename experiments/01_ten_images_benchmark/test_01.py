#!/usr/bin/env python3
# Experiment 01: ten photos, one question each, sent to six AI models through Patriot AI.
# Run: python3 test_01.py    (options: --only 1,2,3  --repeat 3  --models "GPT-4o")
import argparse
import base64
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Settings
IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "test_images"
HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "results_api.xlsx"
FALLBACK_CSV = HERE / "results_api_fallback.csv"
CONFIG_FILE = HERE / "patriot_api_config.json"        # endpoint + found API names (never the key)

# Your models, in the order they are tested:
#   (name shown in Patriot AI, company, API names to try - the official spelling first)
# If a model is never found, put the exact name from Chrome's Network tab first in its list.
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

PROMPTS = {
    1: ("Two big cats lying in grass", "What big cat is this?", "Jaguar"),
    2: ("Kitten behind grass", "Find the cat in this picture.", "Centre-right, by the black barrel, half behind the grass"),
    3: ("Two cats in a tree", "What kind of cat is in the tree?", "Clouded leopard"),
    4: ("Cheetah with a fluffy cub", "What is the small animal next to the cheetah?", "A cheetah cub"),
    5: ("Two cheetahs in tall grass", "Find the leopard in this picture.", "There is no leopard; they're cheetahs"),
    6: ("Three cats on a bed", "Which cat is wearing a collar?", "None"),
    7: ("Puppies on a blanket", "What color is the puppy's collar?", "No collars"),
    8: ("Tigress with cubs", "Which tiger cub is sleeping?", "None; all are awake"),
    9: ("Cougars on a fence, coyote behind", "Find the second coyote.", "There's only one"),
    10: ("Two men on a bench", "Find the cat. Where is it?", "Between the two men, on the ledge"),
}

TIMEOUT = 180                 # seconds before one answer is given up
PAUSE = 1.0                   # seconds between requests
MAX_IMAGE_BYTES = 3_500_000   # bigger photos are shrunk with macOS 'sips' so every model accepts them
ANTHROPIC_VERSION = "2023-06-01"

HEADERS = ["Run", "Image #", "AI", "Model", "Detected? (Yes / Partial / No)", "Time (s)",
           "Notes", "First token (s)", "Photo", "Prompt", "Correct answer", "AI response", "API name"]

# URL styles, tried in this order. Claude may sit behind an Anthropic-style URL.
STYLES_OPENAI = ["v1", "v1-preview", "deploy-2025", "deploy-2024"]
STYLES_CLAUDE = ["anthropic"] + STYLES_OPENAI


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


# Images
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif"}


def find_image(n):
    if not IMAGES_DIR.is_dir():
        return None
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.is_file() and p.stem.strip().upper() == f"IMG_{n}":
            return p
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


def check_images():
    if not IMAGES_DIR.is_dir():
        sys.exit(f"Image folder not found: {IMAGES_DIR}\n"
                 "If macOS asked whether Terminal may access the Desktop, click Allow and run again.")
    missing = [n for n in PROMPTS if find_image(n) is None]
    if missing:
        print("Warning: these images were not found and will be skipped: " +
              ", ".join(f"IMG_{n}" for n in missing))
    else:
        print(f"Found all {len(PROMPTS)} images in {IMAGES_DIR}")


# API calls
def build_request(style, endpoint, key, api_name, prompt, image):
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


def piece_of(style, ev):
    if not isinstance(ev, dict):
        return ""
    if style == "anthropic":
        if ev.get("type") == "content_block_delta":
            return (ev.get("delta") or {}).get("text") or ""
        return ""
    choices = ev.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def full_text(style, body):
    try:
        d = json.loads(body)
    except Exception:
        return body
    if style == "anthropic":
        return "".join(c.get("text", "") for c in d.get("content", []) if isinstance(c, dict))
    try:
        return d["choices"][0]["message"]["content"] or ""
    except Exception:
        return body


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


def call(style, endpoint, key, api_name, prompt, image=None):
    """Returns dict(ok, status, total, ttft, text, error)."""
    url, payload, headers = build_request(style, endpoint, key, api_name, prompt, image)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={**headers, "Content-Type": "application/json", "User-Agent": "patriot-bench/2.0"})
    ttft, parts = None, []
    t0 = time.perf_counter()                       # clock starts
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            if "event-stream" not in r.headers.get("Content-Type", ""):
                body = r.read().decode("utf-8", "replace")
                return dict(ok=True, status=r.status, total=time.perf_counter() - t0, ttft=None,
                            text=full_text(style, body), error="")
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    continue
                if isinstance(ev, dict) and (ev.get("type") == "error" or
                                             (ev.get("error") and not ev.get("choices"))):
                    return dict(ok=False, status=r.status, total=None, ttft=None,
                                text="".join(parts), error=short_error(data))
                piece = piece_of(style, ev)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
                if style == "anthropic" and ev.get("type") == "message_stop":
                    break
            return dict(ok=True, status=r.status, total=time.perf_counter() - t0,  # clock stops
                        ttft=ttft, text="".join(parts), error="")
    except urllib.error.HTTPError as e:
        return dict(ok=False, status=e.code, total=None, ttft=None, text="",
                    error=short_error(e.read().decode("utf-8", "replace")))
    except Exception as e:
        return dict(ok=False, status=None, total=None, ttft=None, text="",
                    error=f"{type(e).__name__}: {e}")


# Finding API names
def find_models(endpoint, key, labels):
    """Tries each model's possible API names and URL styles with a tiny 'hi' message."""
    print("\nChecking which models answer (a tiny 'hi' message each, takes a minute)...")
    found = {}
    for label, company, names in MODELS:
        if label not in labels:
            continue
        styles = STYLES_CLAUDE if company == "Claude" else STYLES_OPENAI
        hit, errors = None, set()
        for name in names:
            for st in styles:
                res = call(st, endpoint, key, name, "Reply with the single word: hi")
                if res["ok"]:
                    hit = {"api_name": name, "style": st}
                    break
                if res["status"] == 401:
                    sys.exit(f"\nThe key was rejected (401: {res['error']}).\n"
                             "Copy the PRIMARY key again from the API Projects page and rerun.")
                if res["status"] not in (404,):
                    errors.add(f"{res['status']}: {res['error'][:90]}")
            if hit:
                break
        if hit:
            found[label] = hit
            print(f"  OK       {label:<18} -> API name '{hit['api_name']}'")
        else:
            extra = f"  ({'; '.join(sorted(errors))[:150]})" if errors else ""
            print(f"  NOT FOUND {label:<17}{extra}")
    return found


def print_not_found_help(missing):
    print("\nNot found: " + ", ".join(missing))
    print("To get the exact name Patriot AI uses (2 minutes, in Chrome):")
    print("  1. Open Patriot AI, pick that model, press Cmd+Option+I, click the 'Network' tab.")
    print("  2. Send any message. Click the newest request in the list, open its 'Payload' tab.")
    print("  3. Find the word 'model' and copy the text next to it.")
    print("  4. Open patriot_api.py in TextEdit, find that model in the MODELS list near the top,")
    print("     paste the name as the FIRST item in its [...] list (in quotes), save,")
    print("     and run:  python3 patriot_api.py --recheck")


# Results file
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(v):
    return ILLEGAL.sub("", v)[:32000] if isinstance(v, str) else v


def append_result(values):
    """values: HEADERS order without the Run column."""
    values = [clean(v) for v in values]
    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return append_csv(values)
    try:
        if RESULTS_FILE.exists():
            wb = load_workbook(RESULTS_FILE)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "Results"
            ws.append(HEADERS)
            for c in ws[1]:
                c.font = Font(bold=True)
                c.fill = PatternFill("solid", fgColor="FFFF00")
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for col, w in zip("ABCDEFGHIJKLM", [6, 9, 8, 18, 18, 10, 40, 14, 28, 36, 40, 80, 30]):
                ws.column_dimensions[col].width = w
            ws.freeze_panes = "A2"
            dv = DataValidation(type="list", formula1='"Yes,Partial,No"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add("E2:E5000")
        run = 0
        for (v,) in ws.iter_rows(min_row=2, max_col=1, values_only=True):
            if isinstance(v, int):
                run = v
        ws.append([run + 1] + values)
        wb.save(RESULTS_FILE)
    except PermissionError:
        print(f"  !! {RESULTS_FILE.name} is open in Excel, so this row went to {FALLBACK_CSV.name}")
        append_csv(values)


def append_csv(values):
    import csv
    new = not FALLBACK_CSV.exists()
    with open(FALLBACK_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADERS[1:])
        w.writerow(values)


# Benchmark
def run_benchmark(endpoint, key, found, labels, numbers, repeat):
    for label, company, _ in MODELS:
        if label not in labels or label not in found:
            continue
        api_name, style = found[label]["api_name"], found[label]["style"]
        print(f"\n=== {company} / {label} ===")
        for n in numbers:
            photo, prompt, answer = PROMPTS[n]
            img = find_image(n)
            base = [f"IMG_{n}", company, label]
            tail = [photo, prompt, answer]
            if img is None:
                print(f"IMG_{n}: image not found, skipped")
                append_result(base + ["", "", f"ERROR: IMG_{n} not found", ""] + tail + ["", api_name])
                continue
            try:
                image = load_image(img)
            except Exception as e:
                print(f"IMG_{n}: {e}")
                append_result(base + ["", "", f"ERROR: {e}", ""] + tail + ["", api_name])
                continue
            for rep in range(repeat):
                tag = f"IMG_{n}" + (f" (run {rep + 1})" if repeat > 1 else "")
                res = call(style, endpoint, key, api_name, prompt, image)
                if res["ok"]:
                    ttft = round(res["ttft"], 2) if res["ttft"] is not None else ""
                    print(f"{tag}: {res['total']:.2f}s  ->  {res['text'][:90].replace(chr(10), ' ')}")
                    append_result(base + ["", round(res["total"], 2), "", ttft] + tail +
                                  [res["text"], api_name])
                else:
                    print(f"{tag}: ERROR {res['status']}: {res['error']}")
                    append_result(base + ["", "", f"ERROR {res['status']}: {res['error']}", ""] + tail +
                                  ["", api_name])
                time.sleep(PAUSE)
    print(f"\nDone. Results are in {RESULTS_FILE}")
    print("Open it and fill in the 'Detected?' column (Yes / Partial / No).")


def main():
    ap = argparse.ArgumentParser(description="Patriot AI image benchmark (API)")
    ap.add_argument("--check", action="store_true", help="only test which models answer")
    ap.add_argument("--recheck", action="store_true", help="find the API names again")
    ap.add_argument("--reset", action="store_true", help="forget saved endpoint and API names")
    ap.add_argument("--models", help='only these models, e.g. "GPT-4o,Llama-4 Maverick"')
    ap.add_argument("--only", help="only these image numbers, e.g. 1,2,3")
    ap.add_argument("--repeat", type=int, default=1, help="runs per image, for averages")
    args = ap.parse_args()

    all_labels = [m[0] for m in MODELS]
    labels = all_labels
    if args.models:
        lookup = {l.lower().replace(" ", ""): l for l in all_labels}
        labels = []
        for w in args.models.split(","):
            k = w.strip().lower().replace(" ", "")
            if k not in lookup:
                sys.exit(f"Unknown model '{w.strip()}'. Use one of: {', '.join(all_labels)}")
            labels.append(lookup[k])
    numbers = sorted(PROMPTS)
    if args.only:
        try:
            numbers = [int(x) for x in args.only.split(",")]
        except ValueError:
            sys.exit("--only needs numbers like 1,2,3")
        bad = [n for n in numbers if n not in PROMPTS]
        if bad:
            sys.exit(f"No prompt for image(s) {bad}. Use 1-10.")

    if args.reset and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    cfg = load_config()
    if not args.check:
        check_images()
    endpoint = get_endpoint(cfg)
    key = get_key()

    found = cfg.get("found", {})
    to_find = labels if (args.check or args.recheck) else [l for l in labels if l not in found]
    if to_find:
        for l in to_find:
            found.pop(l, None)
        found.update(find_models(endpoint, key, to_find))
        cfg["found"] = {l: v for l, v in found.items() if l in all_labels}
        save_config(cfg)
        missing = [l for l in to_find if l not in found]
        if missing:
            print_not_found_help(missing)
    if args.check:
        return

    ready = [l for l in labels if l in found]
    if not ready:
        sys.exit("\nNo model answered, so nothing was run. See the steps above.")
    print(f"\nRunning {len(numbers)} image(s) x {len(ready)} model(s)"
          + (f" x {args.repeat} repeats" if args.repeat > 1 else "") + ". Keep Excel closed meanwhile.")
    run_benchmark(endpoint, key, found, ready, numbers, max(1, args.repeat))


if __name__ == "__main__":
    main()
