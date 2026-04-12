# EE4016 MoodMirror

This repository contains the code and documentation for the EE4016 Group 10 project: **MoodMirror – Lightweight Facial Expression Recognition with Convolutional Neural Networks**.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/muya17/EE4016_MoodMirror.git
   cd EE4016_MoodMirror
   ```
2. **Create your own branch**
   ```bash
   git checkout -b <your-branch-name>
   # Example: git checkout -b feature-01
   ```
3. **Push your branch to GitHub**
   ```bash
   git push --set-upstream origin <your-branch-name>
   ```
4. **Work on your changes, commit, and push**
   ```bash
   git add .
   git commit -m "Describe your changes"
   git push
   ```
5. **Open a Pull Request**
   - Go to GitHub and open a pull request from your branch to `main`.

## Collaboration Guidelines
- **Branching:** If you are contributing, please create your own branch for your work. Do not commit directly to the main branch.
- **Workflow:**
  1. Create a new branch for your feature or fix.
  2. Work on your changes in that branch.
  3. Open a pull request to merge your branch into main after review.
- **Communication:** Use GitHub issues and pull requests for discussion and code review.

## Project Overview
MoodMirror is a lightweight facial expression recognition system that classifies seven emotions (angry, disgust, fear, happy, sad, surprise, neutral) from face images using CNNs. The project includes:
- Baseline models (HOG+SVM, SimpleCNN)
- Proposed LiteCNN architecture
- Ablation studies on regularization and optimization
- Interactive web demo (image upload and webcam capture)

See `EE4016_Project_Proposal.md` for full details.

## Dataset
We use the **FER2013** facial expression dataset for training and evaluation.

- Kaggle dataset page: https://www.kaggle.com/datasets/msambare/fer2013
- You may need a Kaggle account to download the dataset files.

## Run Demo App
1. From the project root, use the one-command run script:
   ```bash
   bash run_app.sh
   ```

2. Manual equivalent (create/open venv, install, run app):
   ```bash
   cd /home/muya/Desktop/EE4016/EE4016_MoodMirror
   if [ ! -d .venv ]; then python3 -m venv .venv; fi
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m streamlit run demo_app.py --server.headless true --server.port 8502
   ```

Live video mode requires browser camera permissions and uses `streamlit-webrtc`.

## Models Used In App Integration
- HOG+SVM artifact: `saved_models/hog_svm_artifact.pkl`
- SimpleCNN checkpoint: `saved_models/SimpleCNN_best.pth`
- LiteCNN checkpoint: `saved_models/LiteCNN_best.pth`

Note: CLCM is now also wired into `demo_app.py` as an optional model, loaded from `saved_models/clcm_best_weights.pth` when present.

## Demo Preview
The Streamlit integration app is ready for team integration.

![MoodMirror Demo UI](assets/demo_preview.png)
