# 01 · Ten images benchmark

The first test. Ten photos from `test_images/`, each with one question, go to six AI models through
Patriot AI: Llama-4 Maverick, GPT-5.4, GPT-4.1, GPT-4o, Claude Sonnet 4.6 and Claude Sonnet 4.5.
The script records each answer, the total time and the time to the first word, and leaves a column in
Excel to mark each answer right or wrong by hand.

```
python3 test_01.py
python3 test_01.py --only 1,2,3 --repeat 3
```

The results of this run weren't kept.
