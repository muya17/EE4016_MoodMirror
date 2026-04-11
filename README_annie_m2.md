# 📂 Data & Baseline Module (M2: Annie)

Hello Team! 
This document outlines the foundation of our `MoodMirror` data pipeline and the classical Machine Learning baseline (HOG+SVM) completed for Week 6 & 7 deliverables.

## 1. Directory Structure Explained
To prevent code conflicts and keep our repo organized, I have set up the following directory structure for our workflow: 

```text
MoodMirror/
├── data/               # [DO NOT PUSH] Place fer2013.csv here.
├── src/                # All core source code goes here.
│   ├── data_loader.py  # [Done] Data parsing and PyTorch DataLoader.
│   └── baselines.py    # [Done] Classical ML baseline (HOG+SVM).
├── requirements.txt    # [Done] Shared Python environment dependencies.
└── README_annie_m2.md  # This document.
```
Feel free to reach out to me if further structure extension or modification needed. 

## 2. Environment Setup
Before running the code, please ensure you have the correct environment:

```Bash
pip install -r requirements.txt
```

## 3. How to Use data_loader.py (For M3 & M4)
The FER2013Dataset correctly parses the raw string pixels into 48x48 matrices. It also includes the exact data augmentations we proposed (Random Horizontal Flip, Rotation, ColorJitter) for the training set only.

To use my dataloader in your CNN training scripts:

```Python
from data_loader import get_dataloaders

# This will return fully prepared PyTorch DataLoaders
train_loader, val_loader, test_loader = get_dataloaders("../data/fer2013.csv", batch_size=64)
```

## 4. How to Run the HOG+SVM Baseline
The baseline extracts HOG features (orientations=9, 8x8 pixels per cell) and trains a LinearSVC.
You can run it directly from the root directory:

```Bash
python src/baselines.py
```
Outputs:
- Prints Test Accuracy and Macro F1-score to the console.
- Automatically saves a confusion_matrix_svm.png in the current directory.