# EE4016 Group 10 — MoodMirror Source Code

**GitHub Repository:** https://github.com/muya17/EE4016_MoodMirror

## Dataset
FER2013 is required to run training scripts. Download from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

Place the dataset under:
```
data/fer2013/
    train/  (angry, disgust, fear, happy, neutral, sad, surprise)
    test/   (same structure)
```

## Folder Structure

| Folder | Contents |
|---|---|
| `01_hog_svm/` | HOG feature extraction + LinearSVC baseline |
| `02_simple_cnn/` | SimpleCNN architecture and training |
| `03_lite_cnn/` | LiteCNN architecture and training |
| `04_clcm/` | CLCM replication baseline |
| `05_ablation_studies/` | E1–E5 ablation experiments (configs, runner, logger) |
| `06_web_app/` | Streamlit demo application |
| `saved_models/` | Pre-trained model weights (.pth) and HOG+SVM artifact (.pkl) |

## Running the Web App
```bash
pip install -r 06_web_app/requirements.txt
streamlit run 06_web_app/demo_app.py
```

## Pre-trained Models
All trained weights are in `saved_models/`:
- `LiteCNN_best.pth` — LiteCNN (1.5M params, 65.88% test accuracy)
- `SimpleCNN_best.pth` — SimpleCNN baseline
- `clcm_best_weights.pth` — CLCM replication (60.39% test accuracy)
- `hog_svm_artifact.pkl` — HOG+SVM classifier (42.83% test accuracy)
