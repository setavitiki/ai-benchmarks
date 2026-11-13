#!/usr/bin/env python3
# Experiment 06: the same questions for three YOLO tools.
# Run: python3 yolo_test.py --prepare once, then python3 yolo_test.py --rounds 30
import argparse
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

from test_06 import IMAGE, IMAGES_DIR, QUESTIONS

HERE = Path(__file__).resolve().parent
DEVICE = "cpu"
ROUNDS = 30
CUTOFFS = [0.25, 0.10, 0.05]
DEFAULT_CONF = 0.25                # the library default: used for the timed rounds
LOWEST = 0.01                      # one untimed prediction saves everything above 1%; cut-offs applied after
WORDS = ["jaguar", "leopard", "tiger", "eye", "paw", "collar", "tree branch", "grass", "snow"]
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}  # COCO

# The fixed rules: question -> (how YOLO answers it, YOLO's correct answer)
RULES = {
    "Q1": ("Top label from YOLO26n-cls", "jaguar"),
    "Q2": ('"yes" if YOLOE finds a "leopard"', "no"),
    "Q3": ("Number of animal boxes from YOLO26n (COCO has no jaguar class)", "2"),
    "Q4": ('Number of "eye" boxes from YOLOE', "3"),
    "Q5": ('Number of "paw" boxes from YOLOE', "0"),
    "Q6": ('"branch" if a YOLOE "tree branch" box sits higher than every animal box', "branch"),
    "Q8": ('"none" unless YOLOE finds a "collar"', "none"),
    "Q10": ('"yes" if YOLOE finds both a "jaguar" and a "tiger"', "no"),
    "Q11": ('"yes" if YOLOE finds "grass"', "yes"),
    "Q12": ('"yes" if YOLO26n finds a "person"', "no"),
    "Q18": ('Whichever of "grass" / "snow" YOLOE finds with higher confidence', "grass"),
    "Q19": ("\"yes\" if YOLO26n finds two animal boxes at the same height (side by side)", "yes"),
    "Q24": ('"yes" if YOLOE finds both a "jaguar" and a "tree branch"', "yes"),
}


def find_image(name):
    path = Path(name).expanduser()
    if not path.is_file() and IMAGES_DIR.is_dir():
        path = next((f for f in sorted(IMAGES_DIR.iterdir())
                     if f.is_file() and f.stem.upper() == name.upper()), path)
    if not path.is_file():
        sys.exit(f"Image '{name}' not found (looked in {IMAGES_DIR}).")
    return path


def load_models():
    try:
        from ultralytics import YOLO, YOLOE
    except ImportError:
        sys.exit("Ultralytics is not installed. Run:  pip3 install ultralytics")
    models = {"YOLO26n (detect, COCO)": YOLO("yolo26n.pt"),
              "YOLO26n-cls (classify, ImageNet)": YOLO("yolo26n-cls.pt")}
    try:
        yoloe = YOLOE("yoloe-26s-seg.pt")
        yoloe.set_classes(WORDS)
    except Exception as e:
        sys.exit(f"YOLOE could not start ({type(e).__name__}: {e}).\n"
                 "Run:  pip3 install git+https://github.com/ultralytics/CLIP.git")
    models["YOLOE-26s (open vocabulary)"] = yoloe
    return models


def predict(model, image, conf=DEFAULT_CONF):
    """One prediction. Returns the three stage times (ms) and what was found."""
    r = model.predict(str(image), device=DEVICE, conf=conf, verbose=False)[0]
    if r.probs is not None:
        found = [(r.names[i], round(float(r.probs.data[i]), 4), None) for i in r.probs.top5]
    else:
        found = [(r.names[int(c)], round(p, 4), [round(v) for v in box])
                 for c, p, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return {k: r.speed[k] for k in ("preprocess", "inference", "postprocess")}, found


def answers(found, cut):
    """Turns the boxes into answers with the fixed RULES, keeping only boxes at or above the cut-off."""
    det = [f for f in found["YOLO26n (detect, COCO)"] if f[1] >= cut]
    ye = [f for f in found["YOLOE-26s (open vocabulary)"] if f[1] >= cut]
    top = found["YOLO26n-cls (classify, ImageNet)"][0][0]
    animals = [f for f in det if f[0] in ANIMALS]

    def ye_has(word):
        return [f for f in ye if f[0] == word]

    def centre_y(f):
        return (f[2][1] + f[2][3]) / 2

    branch = ye_has("tree branch")
    above = bool(branch and animals and min(map(centre_y, branch)) < min(map(centre_y, animals)))
    grass = max((f[1] for f in ye_has("grass")), default=0)
    snow = max((f[1] for f in ye_has("snow")), default=0)
    side_by_side = any(a[2][1] < b[2][3] and b[2][1] < a[2][3]
                       for i, a in enumerate(animals) for b in animals[i + 1:])
    return {
        "Q1": top,
        "Q2": "yes" if ye_has("leopard") else "no",
        "Q3": str(len(animals)),
        "Q4": str(len(ye_has("eye"))),
        "Q5": str(len(ye_has("paw"))),
        "Q6": "branch" if above else ("branch not above" if branch else "nothing found"),
        "Q8": "collar found" if ye_has("collar") else "none",
        "Q10": "yes" if ye_has("jaguar") and ye_has("tiger") else "no",
        "Q11": "yes" if grass else "no",
        "Q12": "yes" if any(f[0] == "person" for f in det) else "no",
        "Q18": "grass" if grass > snow else ("snow" if snow else "neither found"),
        "Q19": "yes" if side_by_side else "no",
        "Q24": "yes" if ye_has("jaguar") and branch else "no",
    }


def prepare(image):
    print("Loading the 3 models (the first time downloads them, including YOLOE's word encoder)...")
    models = load_models()
    for name, model in models.items():
        predict(model, image)                                    # first call warms up
        speed, found = predict(model, image)
        print(f"\nOK  {name}\n    stage times (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in speed.items()))
        print("    found: " + (", ".join(f"{l} {p:.0%}" for l, p, _ in found[:6]) or "nothing"))
    print("\nEverything works. Next:  python3 yolo_test.py --rounds 3")


def run(image, rounds):
    models = load_models()
    found = {}
    for name, model in models.items():
        predict(model, image)                                    # warm-up, not recorded
        found[name] = predict(model, image, conf=LOWEST)[1]      # everything above 1%, for the answers
    timing, first, same = [], {}, {name: True for name in models}
    for rnd in range(1, rounds + 1):                             # timed rounds at the default cut-off
        for name in random.sample(list(models), len(models)):
            speed, out = predict(models[name], image)
            timing.append(dict(round=rnd, model=name, **speed))
            first.setdefault(name, out)
            if [(l, p) for l, p, _ in out] != [(l, p) for l, p, _ in first[name]]:
                same[name] = False                               # YOLO should repeat itself exactly
        print(f"Round {rnd}/{rounds} done")
    return timing, found, same


def write_excel(path, image_name, rounds, timing, found, same):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    bold, fill = Font(bold=True), PatternFill("solid", fgColor="FFFF00")
    green, red = PatternFill("solid", fgColor="E5F4E4"), PatternFill("solid", fgColor="FBEBEB")
    wb = Workbook()

    def header(ws, cells, widths):
        ws.append(cells)
        for c in ws[ws.max_row]:
            c.font, c.fill = bold, fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = f"A{ws.max_row + 1}"

    by_cut = {cut: answers(found, cut) for cut in CUTOFFS}
    wa = wb.active
    wa.title = "Answers"
    wa.append([f"Image {image_name} | YOLO answers through fixed rules | cut-offs 25% (default), 10%, 5%"])
    wa["A1"].font = bold
    cols = ["Q", "Question", "Category", "How YOLO answers (rule)", "Correct answer"]
    for cut in CUTOFFS:
        cols += [f"Answer at {cut:.0%}", f"Correct at {cut:.0%}?"]
    header(wa, cols, [5, 45, 16, 50, 16] + [14, 11] * len(CUTOFFS))
    for q in QUESTIONS:
        if q["id"] not in RULES:
            wa.append([q["id"], q["text"], q["category"], "YOLO cannot answer this", q["answer"]])
            continue
        rule, correct = RULES[q["id"]]
        row = [q["id"], q["text"], q["category"], rule, correct]
        for cut in CUTOFFS:
            got = by_cut[cut][q["id"]]
            row += [got, "yes" if got == correct else "no"]
        wa.append(row)
        for c in wa[wa.max_row][6::2]:
            c.fill = green if c.value == "yes" else red
    total = ["", "Correct out of 13 answerable questions", "", "", ""]
    for cut in CUTOFFS:
        total += ["", sum(by_cut[cut][k] == RULES[k][1] for k in RULES)]
    wa.append(total)
    for c in wa[wa.max_row]:
        c.font = bold

    wt = wb.create_sheet("Timing")
    wt.append([f"Stage times in milliseconds, {rounds} rounds on the {DEVICE.upper()} at the default "
               f"{DEFAULT_CONF:.0%} cut-off (warm-up not counted)"])
    wt["A1"].font = bold
    header(wt, ["Model", "Stage", "Median (ms)", "Fastest (ms)", "Slowest (ms)", "Same answer every round"],
           [34, 14, 12, 12, 12, 14])
    for name in found:
        rows = [t for t in timing if t["model"] == name]
        for stage in ("preprocess", "inference", "postprocess", "total"):
            vals = [t["preprocess"] + t["inference"] + t["postprocess"] if stage == "total" else t[stage]
                    for t in rows]
            wt.append([name, stage, round(statistics.median(vals), 2), round(min(vals), 2),
                       round(max(vals), 2), ("yes" if same[name] else "NO") if stage == "total" else ""])

    wr = wb.create_sheet("Rounds")
    header(wr, ["Round", "Model", "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)", "Total (ms)"],
           [7, 34, 14, 14, 14, 12])
    for t in timing:
        wr.append([t["round"], t["model"], round(t["preprocess"], 2), round(t["inference"], 2),
                   round(t["postprocess"], 2), round(t["preprocess"] + t["inference"] + t["postprocess"], 2)])

    wd = wb.create_sheet("Detections")
    header(wd, ["Model", "Found", "Confidence", "Box (x1, y1, x2, y2)"], [34, 16, 12, 26])
    for name, items in found.items():
        for label, conf, box in sorted(items, key=lambda f: -f[1]):
            wd.append([name, label, conf, str(box) if box else "(whole image)"])
    wb.save(path)
    return by_cut


def main():
    ap = argparse.ArgumentParser(description="The 24-question test for YOLO")
    ap.add_argument("--prepare", action="store_true", help="download the models and check they work")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--image", default=IMAGE, help="image name in the images folder, or a file path")
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("This needs openpyxl. Install it with:  pip3 install openpyxl")
    image = find_image(args.image)
    if args.prepare:
        return prepare(image)

    rounds = max(1, args.rounds)
    timing, found, same = run(image, rounds)
    path = HERE / f"yolo_results_{datetime.now():%Y-%m-%d_%H%M}.xlsx"
    by_cut = write_excel(path, image.name, rounds, timing, found, same)

    print(f"\n{'Q':<5}{'Correct':<10}" + "".join(f"{f'at {c:.0%}':<16}" for c in CUTOFFS))
    for k, (_, correct) in RULES.items():
        print(f"{k:<5}{correct:<10}" + "".join(
            f"{by_cut[c][k] + (' ok' if by_cut[c][k] == correct else ' x'):<16}" for c in CUTOFFS))
    print("Correct of 13: " + ", ".join(f"{sum(by_cut[c][k] == v[1] for k, v in RULES.items())} at {c:.0%}"
                                        for c in CUTOFFS))
    print("\nMedian stage times (ms):")
    for name in found:
        rows = [t for t in timing if t["model"] == name]
        print(f"  {name:<34}" + "  ".join(f"{s} {statistics.median(t[s] for t in rows):.1f}"
                                         for s in ("preprocess", "inference", "postprocess")))
    if not all(same.values()):
        print("WARNING: a model gave different output between rounds (see Timing sheet).")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
