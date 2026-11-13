#!/usr/bin/env python3
# Answers the same questions with four pre-trained YOLO tools.
# Run: python3 yolo_tools_test.py --prepare once, then python3 yolo_tools_test.py --rounds 30
import argparse
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent.parent / "test_images" / "IMG_1.jpg"      # Projects/test_images/IMG_1.jpg
CUTOFF = 0.25
ROUNDS = 30

# The 24 questions and expected answers (the same as the AI models test).
QUESTIONS = [
    ("Q1", "Recognition", "What animal is in this image?", "jaguar"),
    ("Q2", "Existence", "Is there a leopard in this image?", "no"),
    ("Q3", "Counting", "How many jaguars are in this image?", "2"),
    ("Q4", "Counting", "How many eyes are visible in this image?", "3"),
    ("Q5", "Counting", "How many paws are visible in this image?", "0"),
    ("Q6", "Position", "What is above the jaguars?", "branch"),
    ("Q7", "Reasoning", "Are the jaguars awake?", "yes"),
    ("Q8", "Hallucination", "What color is the jaguar's collar?", "none"),
    ("Q9", "Hallucination", "Is the cat in this image white or black?", "neither"),
    ("Q10", "Logic", "Is there a jaguar and a tiger in this image?", "no"),
    ("Q11", "Existence", "Is there grass in this image?", "yes"),
    ("Q12", "Hallucination", "Is there a person in this image?", "no"),
    ("Q13", "Color", "Are the jaguars' spots black?", "yes"),
    ("Q14", "Color", "Is the jaguars' fur mainly grey?", "no"),
    ("Q15", "Attribute", "Is the jaguars' fur striped or spotted?", "spotted"),
    ("Q16", "Activity", "Are the jaguars lying down?", "yes"),
    ("Q17", "Activity", "Are the jaguars running?", "no"),
    ("Q18", "Scene", "Are the jaguars on grass or on snow?", "grass"),
    ("Q19", "Spatial relation", "Are the two jaguars lying next to each other?", "yes"),
    ("Q20", "Comparison", "Are both animals the same species?", "yes"),
    ("Q21", "Knowledge", "On which continent do these animals live in the wild?", "America"),
    ("Q22", "Image style", "Is this image a photograph or a drawing?", "photograph"),
    ("Q23", "Text (OCR)", "Is there any text or writing in this image?", "no"),
    ("Q24", "Logic", "Is there a jaguar and a tree branch in this image?", "yes"),
]

# How YOLO answers each question it can handle, and the types of object a tool must know to answer it.
RULES = {
    "Q1": ("Name of the animal it is most sure of", ["animal"]),
    "Q2": ('"yes" if it finds a leopard', ["leopard"]),
    "Q3": ("Number of animals it finds", ["animal", "boxes"]),
    "Q4": ("Number of eyes it finds", ["eye"]),
    "Q5": ("Number of paws it finds", ["paw"]),
    "Q6": ('"branch" if a branch or tree is found higher than the animals', ["branch", "animal", "boxes"]),
    "Q8": ('"none" unless it finds a collar', ["collar"]),
    "Q10": ('"yes" if it finds both a jaguar and a tiger', ["jaguar", "tiger"]),
    "Q11": ('"yes" if it finds grass', ["grass"]),
    "Q12": ('"yes" if it finds a person', ["person"]),
    "Q18": ("Grass or snow, whichever it is more sure of", ["grass", "snow"]),
    "Q19": ('"yes" if two animals are found side by side', ["animal", "boxes"]),
    "Q24": ('"yes" if it finds both a jaguar and a branch or tree', ["jaguar", "branch"]),
}

COCO_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
OI_ANIMALS = {"Jaguar (Animal)", "Leopard", "Tiger", "Cheetah", "Lion", "Cat", "Dog", "Fox", "Bear", "Raccoon"}
YOLOE_WORDS = ["jaguar", "leopard", "tiger", "eye", "paw", "collar", "tree branch", "grass", "snow", "person"]

# Each tool: file, how it works, what it was trained on, and which of its object types mean what.
TOOLS = {
    "YOLO26n-cls": dict(
        file="yolo26n-cls.pt", kind="classify",
        trained="ImageNet: about 1.28 million photos, each labelled with one of 1,000 types",
        types={"animal": "*", "jaguar": {"jaguar"}, "leopard": {"leopard", "snow_leopard"}, "tiger": {"tiger"}}),
    "YOLO26n (COCO)": dict(
        file="yolo26n.pt", kind="detect",
        trained="COCO: about 118,000 photos with boxes drawn around 80 everyday types of object",
        types={"animal": COCO_ANIMALS, "person": {"person"}}),
    "YOLOv8m (Open Images)": dict(
        file="yolov8m-oiv7.pt", kind="detect",
        trained="Open Images V7: about 1.7 million photos with boxes around 600 types of object",
        types={"animal": OI_ANIMALS, "jaguar": {"Jaguar (Animal)"}, "leopard": {"Leopard"}, "tiger": {"Tiger"},
               "person": {"Person", "Man", "Woman", "Boy", "Girl"}, "branch": {"Tree"}}),
    "YOLOE-26s (any word)": dict(
        file="yoloe-26s-seg.pt", kind="yoloe",
        trained="Large image and caption collections; looks for whatever words it is given",
        types={"animal": {"jaguar", "leopard", "tiger"}, "jaguar": {"jaguar"}, "leopard": {"leopard"},
               "tiger": {"tiger"}, "eye": {"eye"}, "paw": {"paw"}, "collar": {"collar"},
               "branch": {"tree branch"}, "grass": {"grass"}, "snow": {"snow"}, "person": {"person"}}),
}
STAGES = {"preprocess": "The photo is resized and turned into numbers",
          "inference": "The neural network looks at the photo (the actual 'seeing')",
          "postprocess": "The raw output is turned into a clean list of what was found"}


def can_answer(tool, qid):
    needs = RULES[qid][1]
    return all(n == "boxes" and tool["kind"] != "classify" or n in tool["types"] for n in needs)


def load(tool):
    from ultralytics import YOLO, YOLOE
    if tool["kind"] == "yoloe":
        model = YOLOE(tool["file"])
        model.set_classes(YOLOE_WORDS)
        return model
    return YOLO(tool["file"])


def look(model, image):
    """One look: the three stage times (ms) and what was found as (type, confidence, box or None)."""
    r = model.predict(str(image), conf=CUTOFF, device="cpu", verbose=False)[0]
    if r.probs is not None:                                         # a classifier: scores for the whole photo
        probs = r.probs.data.tolist()
        found = [(r.names[i], round(p, 3), None) for i, p in enumerate(probs) if p >= CUTOFF]
        found.sort(key=lambda f: -f[1])
    else:
        found = [(r.names[int(c)], round(p, 3), b)
                 for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return {s: r.speed[s] for s in STAGES}, found


def answer(tool, found):
    """Answers every question this tool can handle, from one look's list."""
    types = tool["types"]

    def of(kind):
        names = types.get(kind, set())
        return [f for f in found if names == "*" or f[0] in names]

    def short(name):
        return re.sub(r"\s*\(.*\)", "", name).replace("_", " ").lower()

    def middle(f):
        return (f[2][1] + f[2][3]) / 2

    animals, branch = of("animal"), of("branch")
    grass = max((f[1] for f in of("grass")), default=0)
    snow = max((f[1] for f in of("snow")), default=0)
    if not branch:
        above = "nothing found"
    elif not animals:
        above = "no animal found"
    else:
        above = "branch" if min(map(middle, branch)) < min(map(middle, animals)) else "not above"
    every = {
        "Q1": short(max(animals, key=lambda f: f[1])[0]) if animals else "nothing found",
        "Q2": "yes" if of("leopard") else "no",
        "Q3": str(len(animals)),
        "Q4": str(len(of("eye"))),
        "Q5": str(len(of("paw"))),
        "Q6": above,
        "Q8": "collar found" if of("collar") else "none",
        "Q10": "yes" if of("jaguar") and of("tiger") else "no",
        "Q11": "yes" if grass else "no",
        "Q12": "yes" if of("person") else "no",
        "Q18": "grass" if grass > snow else ("snow" if snow else "nothing found"),
        "Q19": "yes" if any(a[2][1] < b[2][3] and b[2][1] < a[2][3]
                            for i, a in enumerate(animals) for b in animals[i + 1:]) else "no",
        "Q24": "yes" if of("jaguar") and branch else "no",
    }
    return {q: a for q, a in every.items() if can_answer(tool, q)}


def is_correct(qid, given):
    expected = dict((q[0], q[3]) for q in QUESTIONS)[qid]
    return given == expected or (qid == "Q1" and "jaguar" in given)


# Excel
class Red:
    def __init__(self, value):
        self.value = value


def add_table(ws, headers, rows, widths=None, title=None, formats=None, filters=False):
    from openpyxl.styles import Alignment, Font, PatternFill
    if title:
        ws.append([title])
        ws.cell(ws.max_row, 1).font = Font(bold=True, size=12)
    ws.append(headers)
    top = ws.max_row
    for c in ws[top]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="FFF2CC")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append([c.value if isinstance(c, Red) else c for c in row])
        for i, c in enumerate(row, 1):
            cell = ws.cell(ws.max_row, i)
            if isinstance(c, Red):
                cell.fill = PatternFill("solid", fgColor="F4CCCC")
            if formats and i in formats and isinstance(cell.value, (int, float)):
                cell.number_format = formats[i]
    for i, w in enumerate(widths or [], 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    if filters:
        ws.auto_filter.ref = f"A{top}:{ws.cell(ws.max_row, len(headers)).coordinate}"
        ws.freeze_panes = f"A{top + 1}"
    ws.append([])


def write_excel(path, image_name, rounds, results):
    from openpyxl import Workbook
    wb = Workbook()
    names = list(results)
    total = {n: [sum(lk["speed"].values()) / 1000 for lk in results[n]] for n in names}   # seconds per look
    given = {n: answer(TOOLS[n], results[n][0]["found"]) for n in names}

    ws = wb.active
    ws.title = "Summary"
    rows = []
    for n in names:
        wrong = [q for q, a in given[n].items() if not is_correct(q, a)]
        rows.append([n, f"{len(given[n]) - len(wrong)} of {len(given[n])}",
                     f"{len(QUESTIONS) - len(given[n])}", ", ".join(wrong) or "-", med(total[n], 4)])
    add_table(ws, ["Tool", "Correct (of the questions it can answer)", "Questions it cannot answer",
                   "Wrong answers", "Median total (s)"], rows, [26, 22, 16, 40, 14],
              title=f"YOLO tools | image {image_name} | {rounds} rounds | each tool looks at the image once per "
                    f"round and answers every question it can from that one look | red = wrong",
              formats={5: "0.000"})
    add_table(ws, ["Tool", f"What it found (at least {CUTOFF:.0%} sure)"],
              [[n, ", ".join(f"{w} {p:.0%}" for w, p, _ in results[n][0]["found"]) or "nothing"] for n in names])
    grid = []
    for qid, cat, text, _ in QUESTIONS:
        cells = []
        for n in names:
            if qid not in given[n]:
                cells.append("cannot answer")
            else:
                t = med(total[n], 4)
                cells.append(t if is_correct(qid, given[n][qid]) else Red(t))
        grid.append([qid, cat, text] + cells)
    add_table(ws, ["Q", "Category", "Question"] + names, grid,
              title="Median total time per question (s): every answer from a tool comes from the same look, so a "
                    "tool's time is the same for all its questions; red = wrong",
              formats={i: "0.000" for i in range(4, 4 + len(names))})

    ws = wb.create_sheet("Answers")
    lines = []
    for n in names:
        for i, lk in enumerate(results[n], 1):
            got = answer(TOOLS[n], lk["found"])
            for qid, cat, text, expected in QUESTIONS:
                if qid in got:
                    ok = is_correct(qid, got[qid])
                    line = [i, n, qid, cat, text, expected, got[qid], "yes" if ok else "no", None,
                            round(sum(lk["speed"].values()) / 1000, 4)]
                    lines.append(line if ok else [Red(v) for v in line])
    add_table(ws, ["Round", "Model", "Q", "Category", "Question", "Expected answer", "Answer given", "Correct?",
                   "First word (s)", "Total (s)"], lines, [7, 22, 5, 15, 45, 16, 16, 10, 10, 10],
              formats={10: "0.000"}, filters=True)

    ws = wb.create_sheet("Questions")
    add_table(ws, ["Q", "Category", "Question", "Expected answer", "How YOLO answers"] + names,
              [[qid, cat, text, expected, RULES[qid][0] if qid in RULES else "No YOLO tool can answer this"] +
               [("yes" if qid in given[n] else "no") for n in names] for qid, cat, text, expected in QUESTIONS],
              [5, 15, 45, 16, 50] + [14] * len(names),
              title="Which questions each tool can answer (it needs a type for every object the question is about)")

    ws = wb.create_sheet("Tools")
    add_table(ws, ["Tool", "File", "Trained on", "Types it knows"],
              [[n, TOOLS[n]["file"], TOOLS[n]["trained"], results[n][0]["n_types"]] for n in names],
              [26, 20, 75, 14], title="The four YOLO tools (all pre-trained by Ultralytics; none trained by us)")
    rows = []
    for n in names:
        for s, what in STAGES.items():
            vals = [lk["speed"][s] for lk in results[n]]
            rows.append([n, s, what, round(statistics.median(vals), 2), round(min(vals), 2), round(max(vals), 2)])
        ms = [t * 1000 for t in total[n]]
        rows.append([n, "total", "One complete look", round(statistics.median(ms), 2), round(min(ms), 2),
                     round(max(ms), 2)])
    add_table(ws, ["Tool", "Stage", "What happens", "Median (ms)", "Fastest (ms)", "Slowest (ms)"], rows,
              title=f"One look split into its stages ({rounds} looks per tool; milliseconds, 1,000 ms = 1 s)")
    wb.save(path)
    return given


def med(values, digits=3):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), digits) if values else None


def main():
    ap = argparse.ArgumentParser(description="The image questions, answered by four YOLO tools")
    ap.add_argument("--prepare", action="store_true", help="download the tools and check they work")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--image", default=str(IMAGE))
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        sys.exit("Please install:  python3 -m pip install ultralytics openpyxl")
    image = Path(args.image).expanduser()
    if not image.is_file():
        sys.exit(f"Image not found: {image}")

    results = {}
    for name, tool in TOOLS.items():
        try:
            model = load(tool)
        except Exception as e:
            sys.exit(f"{name} could not start ({e}).\nFor YOLOE also run:  "
                     "python3 -m pip install git+https://github.com/ultralytics/CLIP.git")
        look(model, image)                                          # warm-up, not recorded
        if args.prepare:
            speed, found = look(model, image)
            print(f"OK  {name:<24} {sum(speed.values()):6.1f} ms   found: "
                  + (", ".join(f"{w} {p:.0%}" for w, p, _ in found[:5]) or "nothing"))
            continue
        looks = []
        for _ in range(args.rounds):
            speed, found = look(model, image)
            looks.append(dict(speed=speed, found=found, n_types=len(model.names)))
        if any([f[:2] for f in lk["found"]] != [f[:2] for f in looks[0]["found"]] for lk in looks):
            print(f"WARNING: {name} found different things in different rounds.")
        results[name] = looks
        print(f"{name:<24} done ({datetime.now():%H:%M:%S})")
    if args.prepare:
        print(f"\nEverything works. Image: {image}\nNext:  python3 yolo_tools_test.py --rounds 30")
        return

    path = HERE / "yolo_tools_results.xlsx"
    if path.exists():
        path = HERE / f"yolo_tools_results_{datetime.now():%Y-%m-%d_%H%M}.xlsx"
    given = write_excel(path, image.name, args.rounds, results)
    print()
    for name in results:
        wrong = [q for q, a in given[name].items() if not is_correct(q, a)]
        print(f"{name:<24} {len(given[name]) - len(wrong):>2} of {len(given[name]):>2} correct   "
              f"{statistics.median(sum(lk['speed'].values()) for lk in results[name]):7.1f} ms   "
              f"wrong: {' '.join(wrong) or '-'}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
