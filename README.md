# AI for Health - Sleep Apnea Detection (SRIP 2026)

## Project Overview
This project detects breathing irregularities (Apnea/Hypopnea) during sleep
using physiological signals: Nasal Airflow, Thoracic Movement, and SpO2.

## Directory Structure
```
Project/
├── Data/               # Raw participant data (AP01–AP05)
├── Visualizations/     # PDF plots per participant
├── Dataset/            # Processed windowed dataset
├── models/             # Trained CNN models per fold
├── scripts/
│   ├── vis.py          # Visualization script
│   ├── create_dataset.py  # Preprocessing & dataset creation
│   └── train_model.py  # Model training & evaluation
├── README.md
├── requirements.txt
└── report.pdf
```

## How to Run

### Step 1 — Visualize signals for a participant
```bash
python scripts/vis.py -name "Data/AP01"
```
Generates a PDF in `Visualizations/` with all 3 signals and event overlays.

### Step 2 — Create dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Applies bandpass filtering (0.17–0.4 Hz), creates 30s windows with 50% overlap,
labels each window, saves as `breathing_dataset.pkl`.

### Step 3 — Train and evaluate
```bash
python scripts/train_model.py -dataset_dir "Dataset"
```
Trains a 1D CNN using Leave-One-Participant-Out cross-validation.
Reports Accuracy, Precision, Recall, and Confusion Matrix per fold.

## Methodology

### Preprocessing
- Bandpass filter: 0.17–0.4 Hz (normal breathing range)
- SpO2 clipped to 70–100% to remove sensor artifacts
- SpO2 upsampled from 4 Hz to 32 Hz to match respiration signals
- Windows: 30 seconds, 50% overlap → shape (960, 3) per window

### Labeling
- Window labeled with event type if >50% overlap with annotated event
- Otherwise labeled as Normal
- Classes: Normal, Hypopnea, Obstructive Apnea, Mixed Apnea

### Model
- 3-layer 1D CNN with BatchNormalization and Dropout
- Class weights applied to handle severe class imbalance
- EarlyStopping on validation loss (patience=3)

### Evaluation
- Leave-One-Participant-Out Cross-Validation (5 folds)
- Metrics: Accuracy, Precision, Recall, Confusion Matrix

## Results Summary
| Fold | Participant | Accuracy | Precision | Recall |
|------|-------------|----------|-----------|--------|
| 1    | AP01        | 87.5%    | 90.2%     | 87.5%  |
| 2    | AP02        | 91.4%    | 87.8%     | 91.4%  |
| 3    | AP03        | 19.5%    | 99.0%     | 19.5%  |
| 4    | AP04        | 88.9%    | 85.2%     | 88.9%  |
| 5    | AP05        | 79.6%    | 63.3%     | 79.6%  |
| **Mean** |         | **73.4%**| **85.1%** | **73.4%** |

## Note on AI Tool Usage
This project was developed with assistance from Claude (Anthropic) for code
 debugging. All code has been reviewed and understood by the author.

## Dependencies
See requirements.txt
