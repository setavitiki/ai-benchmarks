# 06 · Twenty-four questions

24 questions about the jaguar photo in 16 categories, taken from published benchmarks (MME, MMBench,
SEED-Bench, MM-Vet, MMStar, POPE, TDIUC and GQA). All six models, 25 rounds, 3,600 answers, plus each question
once without the photo. Also a first YOLO test with three YOLO tools.

```
python3 test_06.py --rounds 25
python3 yolo_test.py --rounds 30
```

Experiment 07 holds the same AI model results in a simpler layout and is the one to read.
Results: `results_2026-11-13_0220.xlsx` and `yolo_results_2026-11-13_0220.xlsx`.
