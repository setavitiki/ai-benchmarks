# 03 · Timing experiment

An attempt to estimate how long the model spends reading the photo, reading the question and writing the
answer, by comparing four kinds of request that differ in one thing each.

```
python3 test_03.py --model "Llama-4 Maverick" --repeat 10
```

Dropped: through an API these times can only be estimated, not measured. Measuring them needs a model
running on your own computer, as in the ModServe paper (arXiv 2502.00937). Trial results:
`results_timing_v2_breakdown.csv` and `results_timing_v2_runs.csv`.
