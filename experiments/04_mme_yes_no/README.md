# 04 · MME yes/no questions

Ten yes/no questions about the jaguar photo, in pairs with one "yes" and one "no" answer, following the MME
benchmark (Fu et al., 2023, arXiv 2306.13394): existence, counting, position, colour and commonsense.

```
python3 test_04.py --model "Llama-4 Maverick" --rounds 10
```

It showed that yes/no questions alone are too narrow, and that a right answer which doesn't start with
"yes" or "no" was being scored as wrong. Experiment 05 replaced it. Results: `results_mme_*.csv`.
