#!/usr/bin/env python3
# The same pointing questions for two YOLO tools. YOLO draws boxes; the centre of each box is its point.
# Run: python3 yolo_locate_test.py --prepare once, then python3 yolo_locate_test.py --rounds 30
import argparse
import statistics
import sys

from locate_test import (HEIGHT, IMAGE, QUESTIONS, WIDTH, Red, add_table, expected, med, result_path,
                         score, show)

CUTOFF = 0.25
ROUNDS = 30
OI_ANIMALS = {"Jaguar (Animal)", "Leopard", "Tiger", "Cheetah", "Lion", "Cat", "Dog", "Fox", "Bear", "Raccoon"}

# Each tool: model file, the words YOLOE is told to look for, and which of its object types mean what.
TOOLS = {
    "YOLOE-26s (any word)": dict(
        file="yoloe-26s-seg.pt",
        words=["jaguar", "leopard", "tiger", "tree branch", "eye", "nose", "paw", "collar"],
        types={"animal": {"jaguar", "leopard", "tiger"}, "branch": {"tree branch"}, "eye": {"eye"},
               "nose": {"nose"}, "paw": {"paw"}, "leopard": {"leopard"}, "collar": {"collar"}}),
    "YOLOv8m (Open Images)": dict(
        file="yolov8m-oiv7.pt", words=None,
        types={"animal": OI_ANIMALS, "leopard": {"Leopard"}}),
}

# question: (type of object, which boxes give the answer)
RULES = {
    "L1": ("animal", "leftmost box"), "L2": ("animal", "rightmost box"), "L3": ("branch", "most confident box"),
    "L4": ("eye", "rightmost box"), "L5": ("nose", "leftmost box"), "L6": ("animal", "each box"),
    "L7": ("eye", "each box"), "L8": ("paw", "each box"), "L9": ("leopard", "each box"),
    "L10": ("collar", "each box"),
}
STAGES = {"preprocess": "The photo is resized and turned into numbers",
          "inference": "The neural network looks at the photo (the actual 'seeing')",
          "postprocess": "Overlapping duplicate boxes are removed, leaving a clean list"}


def load(tool):
    try:
        from ultralytics import YOLO, YOLOE
    except ImportError:
        sys.exit("Please install:  python3 -m pip install ultralytics")
    if tool["words"]:
        model = YOLOE(tool["file"])
        model.set_classes(tool["words"])
        return model
    return YOLO(tool["file"])


def look(model):
    """One look: the three stage times (ms) and the boxes found as (type, confidence, (x1, y1, x2, y2))."""
    r = model.predict(str(IMAGE), conf=CUTOFF, device="cpu", verbose=False)[0]
    found = [(r.names[int(c)], round(p, 3), tuple(round(v) for v in b))
             for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return {s: r.speed[s] for s in STAGES}, found


def answer(tool, found):
    """The points this tool gives for each question it can answer (an empty list means 'none')."""
    out = {}
    for qid, (kind, pick) in RULES.items():
        if kind not in tool["types"]:
            continue
        boxes = [f for f in found if f[0] in tool["types"][kind]]
        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, _, b in boxes]
        if centres and pick == "leftmost box":
            centres = [min(centres)]
        elif centres and pick == "rightmost box":
            centres = [max(centres)]
        elif centres and pick == "most confident box":
            centres = [centres[max(range(len(boxes)), key=lambda i: boxes[i][1])]]
        out[qid] = centres
    return out


def write_excel(path, results, rounds):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    names = list(results)
    summary, cells = [], {}
    for name, res in results.items():
        given = answer(TOOLS[name], res["found"])
        times = [sum(s.values()) / 1000 for s in res["speeds"]]
        right, wrong, invented, dists = 0, [], 0, []
        for qid, _, _, targets, _ in QUESTIONS:
            if qid not in given:
                cells[name, qid] = "cannot answer"
                continue
            result, dist, _ = score(qid, show(given[qid]) if given[qid] else "none")
            right += result == "correct"
            invented += result == "invented"
            dists.append(dist)
            text = show(given[qid]) if given[qid] else "none"
            text += "" if result == "correct" else " (missed)" if result == "answered none" else f" ({result})"
            cells[name, qid] = text if result == "correct" else Red(text)
            if result != "correct":
                wrong.append(qid)
        summary.append([name, f"{right} of {len(given)}", ", ".join(wrong) or "-", invented, med(dists, 0),
                        med(times, 3)])
    add_table(ws, ["Tool", "Correct (of the questions it can answer)", "Wrong", "Invented",
                   "Median distance from box centre (px)", "Time per look (s)"], summary,
              [26, 20, 30, 10, 16, 12],
              title=f"YOLO pointing | image {IMAGE.name} ({WIDTH} x {HEIGHT} px) | {rounds} rounds | each box's "
                    f"centre is its point | boxes it is at least {CUTOFF:.0%} sure of | red = wrong",
              formats={6: "0.000"})
    add_table(ws, ["Q", "Category", "Question", "Expected answer"] + names,
              [[qid, c, t, expected(tg)] + [cells[n, qid] for n in names] for qid, c, t, tg, _ in QUESTIONS],
              [5, 20, 40, 40] + [40] * len(names), title="Points given (the same in every round); red = wrong")

    ws = wb.create_sheet("What it found")
    add_table(ws, ["Tool", "Object type", "Sure (%)", "Left", "Top", "Right", "Bottom"],
              [[n, w, round(p * 100), *b] for n, res in results.items() for w, p, b in res["found"]],
              [26, 20, 10, 8, 8, 8, 8], title="Every box found in one look (the same in every round)")

    def how(qid):
        kind, pick = RULES[qid]
        return f"Centre of {pick if pick == 'each box' else 'the ' + pick} labelled '{kind}'; no box = none"

    ws = wb.create_sheet("Rules")
    add_table(ws, ["Q", "Question", "How YOLO answers"] + names,
              [[qid, t, how(qid)] +
               ["yes" if RULES[qid][0] in TOOLS[n]["types"] else "cannot answer" for n in names]
               for qid, _, t, _, _ in QUESTIONS], [5, 40, 45] + [22] * len(names))

    ws = wb.create_sheet("Stage times")
    rows = []
    for name, res in results.items():
        for stage, what in STAGES.items():
            vals = [s[stage] for s in res["speeds"]]
            rows.append([name, stage, what, med(vals), round(min(vals), 2), round(max(vals), 2)])
        ms = [sum(s.values()) for s in res["speeds"]]
        rows.append([name, "total", "One complete look", med(ms), round(min(ms), 2), round(max(ms), 2)])
    add_table(ws, ["Tool", "Stage", "What happens", "Median (ms)", "Fastest (ms)", "Slowest (ms)"], rows,
              [26, 12, 60, 12, 12, 12], title=f"One look split into its stages ({rounds} looks, 1,000 ms = 1 s)")
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="The pointing questions, answered by YOLO")
    ap.add_argument("--prepare", action="store_true", help="download the tools and check they work")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Please install openpyxl:  python3 -m pip install openpyxl")
    if not IMAGE.is_file():
        sys.exit(f"Image not found: {IMAGE}")

    results = {}
    for name, tool in TOOLS.items():
        model = load(tool)
        look(model)                                                  # warm-up, not recorded
        speeds, finds = [], []
        for _ in range(1 if args.prepare else args.rounds):
            speed, found = look(model)
            speeds.append(speed)
            finds.append(found)
        if any(f != finds[0] for f in finds):
            print(f"WARNING {name}: found different things in different rounds.")
        results[name] = dict(speeds=speeds, found=finds[0])
        print(f"{name}: {', '.join(f'{w} {p:.0%}' for w, p, _ in finds[0]) or 'nothing'}  "
              f"({statistics.median(sum(s.values()) for s in speeds):.0f} ms per look)")
    if args.prepare:
        print("Both tools work.")
        return
    path = result_path("yolo_locate_results")
    write_excel(path, results, args.rounds)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
