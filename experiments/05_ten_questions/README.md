# 05 · Ten questions

Ten questions about the jaguar photo, covering recognition, counting, position, reasoning, things that aren't
there, and logic. All six models, ten rounds, in random order. Run twice: with an answer instruction
("Answer with one word") and without one.

```
python3 test_05.py --rounds 10
python3 test_05.py --rounds 10 --natural
```

With the instruction: GPT-4.1 88%, Llama-4 Maverick 79%, Claude Sonnet 4.5 79%, GPT-4o 77%,
Claude Sonnet 4.6 70%, GPT-5.4 65%. Without it, answers got longer and slower, counting improved when models
explained their working, and the replies had to be checked by hand.

Results: `results_2026-11-03_0136.xlsx` (with the instruction), `results_2026-11-03_0825_natural.xlsx` (without).
