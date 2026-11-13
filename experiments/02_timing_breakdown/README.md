# 02 · Timing breakdown

One photo (the jaguars) and one question, "What big cat is this?", sent to one model several times. Each
request is split into connecting to the server, uploading, the server's work before the first word, and
the answer arriving. The question is also sent without the photo, to see what the photo adds.

```
python3 test_02.py --model "Llama-4 Maverick" --repeat 3
```

For Llama-4 Maverick the photo counted as about 1,900 tokens, and connecting took about 0.04 s of a 1 to 3 s
total. The upload figure turned out to be unreliable, so later experiments time the response more simply.
Results: `results_timing_fallback.csv`.
