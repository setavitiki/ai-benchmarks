#!/usr/bin/env python3
# Answers the same questions with YOLOE, which looks at the photo once per round.
# Run: python3 yolo_test.py --prepare once, then python3 yolo_test.py --rounds 30
import argparse
import statistics
import sys
from pathlib import Path

from ai_models_test import IMAGE, QUESTIONS, Red, add_table, med, result_path

MODEL, NAME = "yoloe-26s-seg.pt", "YOLOE-26s"
CUTOFF = 0.25
ROUNDS = 30
WORDS = ["jaguar", "leopard", "tiger", "eye", "paw", "collar", "tree branch", "grass", "snow", "person"]
ANIMALS = ("jaguar", "leopard", "tiger")
STAGES = {"preprocess": "The photo is resized to 640 pixels and turned into numbers",
          "inference": "The neural network looks at the photo and finds objects (the actual 'seeing')",
          "postprocess": "Overlapping duplicate boxes are removed, leaving a clean list"}

# question: (how YOLO answers it, YOLO's correct answer)
RULES = {
    "Q1": ("The animal name on its most confident animal box", "jaguar"),
    "Q2": ('"yes" if it finds a leopard', "no"),
    "Q3": ("Number of animal boxes (jaguar, leopard or tiger)", "2"),
    "Q4": ("Number of eye boxes", "3"),
    "Q5": ("Number of paw boxes", "0"),
    "Q6": ('"branch" if a tree-branch box is higher than every animal box', "branch"),
    "Q8": ('"none" unless it finds a collar', "none"),
    "Q10": ('"yes" if it finds both a jaguar and a tiger', "no"),
    "Q11": ('"yes" if it finds grass', "yes"),
    "Q12": ('"yes" if it finds a person', "no"),
    "Q18": ("Whichever of grass or snow it finds with more confidence", "grass"),
    "Q19": ('"yes" if two animal boxes are side by side (at the same height)', "yes"),
    "Q24": ('"yes" if it finds both a jaguar and a tree branch', "yes"),
}


def load_model():
    try:
        from ultralytics import YOLOE
    except ImportError:
        sys.exit("Please install:  python3 -m pip install ultralytics")
    try:
        model = YOLOE(MODEL)
        model.set_classes(WORDS)
        return model
    except Exception as e:
        sys.exit(f"YOLOE could not start ({e}).\n"
                 "Please install:  python3 -m pip install git+https://github.com/ultralytics/CLIP.git")


def look(model, image):
    """One look at the image: the three stage times (ms) and the objects found as (word, confidence, box)."""
    r = model.predict(str(image), conf=CUTOFF, device="cpu", verbose=False)[0]
    found = [(r.names[int(c)], round(p, 3), b)
             for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return {s: r.speed[s] for s in STAGES}, found


def answer(found):
    """Turns one look's list of objects into answers, using RULES."""
    def boxes(*names):
        return [f for f in found if f[0] in names]

    def middle(f):
        return (f[2][1] + f[2][3]) / 2                                # vertical middle of a box

    animals, branch = boxes(*ANIMALS), boxes("tree branch")
    grass = max((f[1] for f in boxes("grass")), default=0)
    snow = max((f[1] for f in boxes("snow")), default=0)
    side_by_side = any(a[2][1] < b[2][3] and b[2][1] < a[2][3]
                       for i, a in enumerate(animals) for b in animals[i + 1:])
    if not branch:
        above = "nothing found"
    elif not animals:
        above = "no animal found"
    else:
        above = "branch" if min(map(middle, branch)) < min(map(middle, animals)) else "branch not above"
    return {
        "Q1": max(animals, key=lambda f: f[1])[0] if animals else "nothing found",
        "Q2": "yes" if boxes("leopard") else "no",
        "Q3": str(len(animals)),
        "Q4": str(len(boxes("eye"))),
        "Q5": str(len(boxes("paw"))),
        "Q6": above,
        "Q8": "collar found" if boxes("collar") else "none",
        "Q10": "yes" if boxes("jaguar") and boxes("tiger") else "no",
        "Q11": "yes" if grass else "no",
        "Q12": "yes" if boxes("person") else "no",
        "Q18": "grass" if grass > snow else ("snow" if snow else "nothing found"),
        "Q19": "yes" if side_by_side else "no",
        "Q24": "yes" if boxes("jaguar") and branch else "no",
    }


def write_excel(path, image_name, rounds, looks):
    from openpyxl import Workbook
    wb = Workbook()
    found = looks[0]["found"]
    totals = [sum(lk["speed"].values()) / 1000 for lk in looks]               # seconds per look
    given = answer(found)
    wrong = [k for k, (_, correct) in RULES.items() if given[k] != correct]

    ws = wb.active
    ws.title = "Summary"
    add_table(ws, ["Model", "Correct", "Wrong answers (times wrong)", "Median first word (s)", "Median total (s)"],
              [[NAME, f"{len(RULES) - len(wrong)} of {len(RULES)} it can answer "
                      f"({len(QUESTIONS) - len(RULES)} it cannot answer)",
                ", ".join(f"{k} ({rounds} of {rounds})" for k in wrong) or "-", "-", med(totals, 4)]],
              [30, 30, 55, 14, 14],
              title=f"YOLO | image {image_name} | {rounds} rounds | YOLO looks at the image once per round and "
                    f"answers every question from that one look | red = answered wrong",
              formats={5: "0.000"})
    ws.append(["What it found (at least 25% sure): " +
               (", ".join(f"{w} {p:.0%}" for w, p, _ in found) or "nothing")])
    ws.append([])
    add_table(ws, ["Q", "Category", "Question", NAME],
              [[q["id"], q["category"], q["text"],
                "cannot answer" if q["id"] not in RULES else
                Red(med(totals, 4)) if q["id"] in wrong else med(totals, 4)] for q in QUESTIONS],
              title="Median total time per question (s); every answer comes from the same look, so the time "
                    "is the same; red = wrong",
              formats={4: "0.000"})

    ws = wb.create_sheet("Answers")
    lines = []
    for i, lk in enumerate(looks, 1):
        got = answer(lk["found"])
        for q in QUESTIONS:
            if q["id"] in RULES:
                ok = got[q["id"]] == RULES[q["id"]][1]
                line = [i, NAME, q["id"], q["category"], q["text"], RULES[q["id"]][1], got[q["id"]],
                        "yes" if ok else "no", None, round(totals[i - 1], 4)]
                lines.append(line if ok else [Red(v) for v in line])
    add_table(ws, ["Round", "Model", "Q", "Category", "Question", "Expected answer", "Answer given",
                   "Correct?", "First word (s)", "Total (s)"], lines, [7, 12, 5, 15, 45, 16, 16, 10, 10, 10],
              formats={10: "0.000"}, filters=True)

    ws = wb.create_sheet("Questions")
    add_table(ws, ["Q", "Category", "Question", "Expected answer", "How YOLO answers"],
              [[q["id"], q["category"], q["text"], RULES[q["id"]][1] if q["id"] in RULES else q["answer"],
                RULES[q["id"]][0] if q["id"] in RULES else "YOLO cannot answer this"] for q in QUESTIONS],
              [5, 16, 50, 26, 60])

    ws = wb.create_sheet("Stage times")
    rows = []
    for stage, what in STAGES.items():
        vals = [lk["speed"][stage] for lk in looks]
        rows.append([stage, what, round(statistics.median(vals), 2), round(min(vals), 2), round(max(vals), 2)])
    ms = [t * 1000 for t in totals]
    rows.append(["total", "One complete look", round(statistics.median(ms), 2), round(min(ms), 2), round(max(ms), 2)])
    add_table(ws, ["Stage", "What happens", "Median (ms)", "Fastest (ms)", "Slowest (ms)"], rows,
              [14, 75, 13, 13, 13],
              title=f"One look at the image, split into its three stages ({rounds} looks, milliseconds: "
                    f"1,000 ms = 1 s)")
    wb.save(path)
    return given, wrong


def main():
    ap = argparse.ArgumentParser(description="The image questions, answered by YOLO")
    ap.add_argument("--prepare", action="store_true", help="download YOLOE and check it works")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--image", default=str(IMAGE))
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Please install openpyxl:  python3 -m pip install openpyxl")
    image = Path(args.image).expanduser()
    if not image.is_file():
        sys.exit(f"Image not found: {image}")

    model = load_model()
    look(model, image)                                                 # warm-up, not recorded
    if args.prepare:
        speed, found = look(model, image)
        print("YOLOE works. One look: " +", ".join(f"{s} {v:.1f} ms" for s, v in speed.items()))
        print("Found: " + (", ".join(f"{w} {p:.0%}" for w, p, _ in found) or "nothing"))
        return

    looks = []
    for i in range(1, args.rounds + 1):
        speed, found = look(model, image)
        looks.append(dict(speed=speed, found=found))
    if any([(w, p) for w, p, _ in lk["found"]] != [(w, p) for w, p, _ in looks[0]["found"]] for lk in looks):
        print("WARNING: YOLO found different things in different rounds.")

    path = result_path("yolo_results")
    given, wrong = write_excel(path, image.name, args.rounds, looks)
    print(f"{len(RULES) - len(wrong)} of {len(RULES)} correct. Wrong: {', '.join(wrong) or '-'}")
    print(f"One look: median {statistics.median(sum(lk['speed'].values()) for lk in looks):.1f} ms")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
