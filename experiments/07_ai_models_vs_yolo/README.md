# 07 · AI models vs YOLO

The main comparison. The same 24 questions about the jaguar photo (`test_images/IMG_1.jpg`) go to six AI
models through Patriot AI (25 rounds) and to YOLOE-26s running on a Mac (30 rounds). The questions cover
16 categories, from recognising the animal to colour, counting, position and things that aren't there.

**Timing.** Each AI question is its own request, with only the photo and that question. The clock starts when
the request is sent and stops at the first word and at the last word. YOLO looks at the photo once per round
and answers everything from that one look, so its answers in a round share one time.

```
python3 ai_models_test.py --rounds 1 --no-image-rounds 0   # trial, about 7 minutes
python3 ai_models_test.py --rounds 25                      # full run, about 3 hours
python3 yolo_test.py --prepare                             # once, downloads YOLOE
python3 yolo_test.py --rounds 30                           # about a minute
```

## Results

| Model | Correct | Median time |
| --- | --- | --- |
| GPT-4.1 | 94.8% | 1.52 s |
| Claude Sonnet 4.5 | 91.0% | 1.48 s |
| Llama-4 Maverick | 89.0% | 1.34 s |
| GPT-4o | 88.8% | 2.89 s |
| Claude Sonnet 4.6 | 87.5% | 1.14 s |
| GPT-5.4 | 85.2% | 1.37 s |
| YOLOE-26s | 5 of the 13 questions it can answer | 0.09 s |

- Every model got 17 of the 24 questions right every time. The differences came from seven: leopard or not,
  counting eyes and paws, what's above the jaguars, the collar, the cat's colour, and whether the jaguars
  lie side by side.
- Wrong answers repeated round after round. Claude Sonnet 4.6 called the branch a rock 25 times out of 25.
- The time hardly changed between questions; it only went up when a model wrote a sentence instead of a word.
- The colour and pattern questions could be answered without the photo (see the Without image sheet), so
  getting them right says little about seeing.
- YOLOE found only "tiger, 42%" in the photo, so its five right answers are all "nothing there" answers.

`ai_models_results.xlsx`: Summary, Answers (wrong answers in red), Questions, Without image.
`yolo_results.xlsx`: Summary, Answers, Questions, Stage times.
