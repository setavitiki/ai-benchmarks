# 08 · YOLO tools

The 24 questions again, this time for four YOLO tools that come already trained. Each tool runs on the Mac's
CPU for 30 rounds and only answers questions about objects it was trained to recognise, so a tool that
doesn't know "collar" skips the collar question.

| Tool | Trained on |
| --- | --- |
| YOLO26n-cls | ImageNet: 1.28 million photos, 1,000 kinds of object, jaguar included |
| YOLO26n | COCO: 118,000 photos, 80 everyday kinds of object, no jaguar |
| YOLOv8m Open Images | Open Images: 1.7 million photos, about 600 kinds of object, jaguar included |
| YOLOE-26s | Photos with captions; searches for any word it's given |

```
python3 yolo_tools_test.py --prepare      # once, downloads the four tools
python3 yolo_tools_test.py --rounds 30
```

## Results

| Tool | Correct | Questions it can answer | Time per photo |
| --- | --- | --- | --- |
| YOLO26n-cls | 3 of 3 ("jaguar, 96%") | 3 | 0.008 s |
| YOLO26n | 3 of 4 (says "cat") | 4 | 0.034 s |
| YOLOv8m Open Images | 5 of 8 (finds both jaguars) | 8 | 0.140 s |
| YOLOE-26s | 5 of 13 (says "tiger") | 13 | 0.093 s |

What decides whether a tool recognises the jaguar is whether jaguars were in its training photos. YOLO is
fast and gives the same answer every time, but it can only answer a few kinds of question. The jaguar photo
may itself come from Open Images, so the Open Images tool may have seen it during training.

`yolo_tools_results.xlsx`: Summary, Answers (wrong answers in red), Questions, Tools.
