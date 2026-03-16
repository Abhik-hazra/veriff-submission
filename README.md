# Veriff — Person Detection Challenge

Classifies each video as `Single Person` or `Multiple People` using YOLOv10.

## Folder Structure

submission/
├── veriff_submission.py  # main script
├── requirements.txt      # dependencies
├── videos/               # place .mp4 files here
└── output/               # results written here after run

## Setup

    pip install -r requirements.txt

## Run

Place all .mp4 videos inside the `videos/` folder, then:

    python veriff_submission.py

## Output

| File | Description |
|---|---|
| `output/results.csv` | Prediction for each video |
| `output/evaluation.csv` | Accuracy vs labels (if labels.txt provided) |
| `output/run.log` | Full execution log |

## Notes

- If `labels.txt` is present, the script evaluates accuracy automatically
- If `labels.txt` is missing, it skips evaluation and still writes predictions
- PyTorch GPU/MPS install: https://pytorch.org/get-started/locally/
