# 09 · Locate objects

Can the AI models say *where* something is in the jaguar photo (`test_images/IMG_1.jpg`, 1024 × 685 pixels),
not just *what* is there? This is called visual grounding.

**Disclaimer.** The models can't mark images. They gave locations as pixel coordinates in text, which were
compared with hand-made reference boxes.

## Pretest

`pretest.py` sent five requests with no rules about the answer format, to see what each model does on its own
(1 round, 30 requests, results in `pretest_results.xlsx`).

- No model could draw on or crop the photo. Three pretended to: GPT-4.1 gave made-up image links, Llama said
  "Here is a cropped version" with nothing attached, and Claude Sonnet 4.6 started "Here is the cropped image"
  before admitting it couldn't.
- Asked "where is…?", five models answered in words ("lower left"). Only GPT-5.4 gave pixel coordinates, so
  the main test asks for pixels.
- Both Claude models said the right jaguar's eye is not visible. It is.
- GPT-5.4 again described a collar that isn't there.

## The test

Ten questions, each ending with the same instruction:

> The image is 1024 pixels wide and 685 pixels tall; (0, 0) is the top-left corner. Answer only with the pixel
> coordinates of the centre, as (x, y). If there are several, give one (x, y) for each. If it is not in the
> image, answer: none.

| Q | Question | Correct answer | Idea taken from |
| --- | --- | --- | --- |
| L1 | Point to the jaguar on the left. | 1 point in box A | RefCOCO (location words) |
| L2 | Point to the jaguar on the right. | 1 point in box B | RefCOCO |
| L3 | Point to the branch above the jaguars. | 1 point in box C | RefCOCO / RefCOCOg (relationship) |
| L4 | Point to the eye of the jaguar on the right. | 1 point in box D | Point-It-Out (object parts) |
| L5 | Point to the nose of the jaguar on the left. | 1 point in box E | Point-It-Out |
| L6 | Point to both jaguars. | 1 point in A and 1 in B | gRefCOCO (several targets) |
| L7 | Point to every visible eye. | 1 point in each of D, F, G | Molmo / PixMo-Points (counting by pointing) |
| L8 | Point to every visible paw. | none | Molmo, POPE |
| L9 | Point to the leopard. | none | gRefCOCO (no target) |
| L10 | Point to the jaguar's collar. | none | gRefCOCO, POPE |

L8–L10 repeat mistakes from experiment 07 (paws, leopard, collar), now asked as "point to it".

**Answer list.** Seven boxes drawn by hand (left, top, right, bottom, in pixels): A jaguar on the left
(0, 235, 495, 545), B jaguar on the right, head and chest (540, 210, 810, 505), C branch (0, 0, 1024, 190),
D right jaguar's eye (722, 288, 754, 316), E left jaguar's nose (380, 420, 450, 452), F and G the left
jaguar's eyes (342, 344, 380, 374) and (440, 372, 474, 400). Box B leaves out the right jaguar's foreleg so
that A and B don't overlap.

**Scoring** is automatic. An answer is correct when it gives exactly one point inside each target box, or
"none" when nothing should be there. Otherwise it's one of: wrong place, wrong number of points, pointed to
something not there, pointed anyway (said it isn't there, then gave coordinates), answered none (when the
thing is there), or unclear (no coordinates and no "none"). A point repeated at the end of a reply counts once,
and number words are read as digits (one reply wrote "(512, ninety)").
The distance from each box centre to the nearest point is recorded as well. Benchmarks such as RefCOCO
compare whole boxes or outlines; checking whether a point lands in a box is the simpler version used for
pointing (Molmo, Point-It-Out).

**Timing** is the same as in experiment 07: the connection is opened first, and the clock runs from sending
the request to the first and to the last word.

**YOLO** draws boxes by design, so the centre of each box is its point. Two tools take part: YOLOE-26s, told
to look for jaguar, leopard, tiger, tree branch, eye, nose, paw and collar, and YOLOv8m trained on Open Images,
which knows jaguars and leopards but not eyes, noses or branches, so it answers only L1, L2, L6 and L9.

```
python3 locate_test.py --rounds 1 --no-image-rounds 0   # trial, 60 requests, about 3 minutes
python3 locate_test.py --rounds 10                       # 600 requests + 60 without the image, about 35 minutes
python3 yolo_locate_test.py --prepare                    # once, downloads the two tools
python3 yolo_locate_test.py --rounds 30
python3 locate_test.py --regrade locate_results_old.xlsx  # score saved replies again, no new requests
```

## Results

10 rounds, 600 questions with the photo and 60 without it. The first scoring counted a point twice when a
model repeated it at the end of its reply; the replies were scored again with `--regrade`, so these are the
corrected numbers.

| Model | Correct | Median distance from the target | Median time |
| --- | --- | --- | --- |
| GPT-5.4 | 79% | 24 px | 1.45 s |
| GPT-4.1 | 57% | 95 px | 1.50 s |
| Llama-4 Maverick | 56% | 95 px | 1.45 s |
| Claude Sonnet 4.6 | 51% | 59 px | 1.10 s |
| Claude Sonnet 4.5 | 49% | 55 px | 3.42 s |
| GPT-4o | 48% | 75 px | 2.98 s |
| YOLOE-26s | 5 of 10 | 117 px | 0.09 s |
| YOLOv8m (Open Images) | 3 of the 4 it can answer | 63 px | 0.14 s |

- **Whole objects are easy, but also easy to guess.** Four models found both jaguars and the branch every
  time; GPT-4.1 often pointed below the jaguars, and GPT-4o put the two jaguars too close together. Without
  the photo, GPT-5.4 and both Claude models guessed points such as (256, 342), the middle of the left half,
  and 11 of those guesses landed in the right box. So these questions say little about seeing.
- **Small parts separate the models.** Only GPT-5.4 found the right jaguar's eye and the left jaguar's nose
  (18 of 20; for the eye, a median of 7 px from its centre). The others pointed elsewhere on the head or
  below it, a median of 38 to 170 px away.
- **Pointing to every eye failed almost always** (1 of 60). Claude Sonnet 4.6 and Llama gave only two eyes,
  one per jaguar. In experiment 07, GPT-5.4, GPT-4.1 and Claude Sonnet 4.5 counted
  "3 eyes" correctly every time; they can count the eyes but not place them.
- **Asked for a leopard, five models pointed at the jaguars.** Only GPT-4.1 answered "none". Both Claude
  models first wrote that these are jaguars, not leopards, then gave coordinates anyway.
- **Both Claude models pointed to paws every time** (20 of 20), though none are visible; GPT-5.4 said "none"
  every time.
- **The collar was hardly invented** (2 of 60). GPT-5.4, which described a collar in all 25 rounds of
  experiment 07 and in the pretest, said "none" every time here. The instruction "if it is not in the image,
  answer: none" seems to make "none" an acceptable answer.
- **Without the photo**, GPT-5.4, Claude Sonnet 4.6 and Claude Sonnet 4.5 gave coordinates for 8 to 10 of the
  10 questions, and Claude Sonnet 4.5 sometimes described a scene that wasn't there (a dog, a giraffe).
  GPT-4.1, GPT-4o and Llama mostly answered "none" or said they couldn't see an image.
- **Pointing took no longer than the questions in experiment 07.** Claude Sonnet 4.5 was slower because it
  explained its reasoning before giving the coordinates.
- **YOLO** answered in under 0.15 s and never pointed to something that isn't there, but it found only what it
  was trained to look for: YOLOE found the left jaguar (calling it a tiger) and its nose, but no eyes or branch;
  the Open Images tool found both jaguars but also labelled the left one "Cheetah", giving three points for two
  jaguars.

**Limits.** One photo, and one person drew the reference boxes. A point just outside a small box counts as
wrong, so the distance column shows how close each answer came. Answers barely changed between rounds, so
10 rounds were enough.

`locate_results.xlsx`: Summary, Distance and time, Answers (wrong answers in red), Questions, Answer list,
Without image. `yolo_locate_results.xlsx`: Summary, What it found, Rules, Stage times.
`pretest_results.xlsx`: the pretest replies.
