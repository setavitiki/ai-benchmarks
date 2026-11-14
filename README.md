# AI benchmarks

How well, and how fast, do AI systems understand a photo? These experiments ask AI chat models
(GPT, Claude, Llama, through the Patriot AI API) and YOLO object detectors the same questions about the
same photos, and record every answer and its time.

## Experiments

| # | Experiment | In short |
| --- | --- | --- |
| 01 | [Ten images benchmark](experiments/01_ten_images_benchmark) | Ten photos, one question each, six AI models |
| 02 | [Timing breakdown](experiments/02_timing_breakdown) | Where the time goes for one photo and one question |
| 03 | [Timing experiment](experiments/03_timing_experiment) | Estimating the model's internal time through the API (dropped) |
| 04 | [MME yes/no](experiments/04_mme_yes_no) | Ten yes/no questions in the style of the MME benchmark |
| 05 | [Ten questions](experiments/05_ten_questions) | Ten questions, six models, ten rounds |
| 06 | [Twenty-four questions](experiments/06_twenty_four_questions) | 24 questions in 16 categories, 25 rounds, plus a first YOLO test |
| 07 | [AI models vs YOLO](experiments/07_ai_models_vs_yolo) | **Main result**: the 24 questions for six AI models and YOLOE |
| 08 | [YOLO tools](experiments/08_yolo_tools) | **Main result**: the 24 questions for four pre-trained YOLO tools |
| 09 | [Locate objects](experiments/09_locate_objects) | Pointing: AI models and YOLO give pixel coordinates, scored against reference boxes |

## What we found

- The AI models got 85% to 95% of the 24 questions right (GPT-4.1 best), taking 1 to 3 seconds per answer.
- Their mistakes repeated every round: counting eyes and paws, mistaking a jaguar for a leopard, and
  describing things that aren't there, like a collar.
- YOLO needs only 8 to 140 milliseconds per photo, but answers just the few questions about objects it was
  trained to find. Tools trained on photos that include jaguars recognised them; the others said "cat" or "tiger".
- Asked to point (experiment 09), the models found whole jaguars easily, but only GPT-5.4 could point to an
  eye or a nose. Most models pointed at the jaguars when asked for a leopard.

## Layout

```
README.md, .gitignore, requirements.txt
test_images/          the photos (IMG_1.jpg is the jaguar photo used from experiment 02 on)
experiments/NN_name/  one folder per experiment, each with its own README, scripts and results
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The AI model experiments need a Patriot AI endpoint and key. The scripts ask for both; the endpoint is
saved in a `patriot_api_config.json` next to the script (kept out of Git) and the key is never saved.

## Adding an experiment

Give it the next number (`10_name`), read photos from `test_images/`, add a README, and when it's finished
tag it: `git tag 10-final && git push --tags`.
