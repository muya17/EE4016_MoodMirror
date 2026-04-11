# Proposal Progress Snapshot

Date: 2026-04-12
Branch: presentation

## What Is Working

- The demo app starts successfully and serves on Streamlit.
- The dataset now loads from the folder layout at `data/fer2013`.
- SimpleCNN and LiteCNN use real checkpoint-backed inference in the demo when checkpoints are present.
- HOG+SVM now has a real exported artifact at `saved_models/hog_svm_artifact.pkl`.
- The core training scripts point at the folder dataset and can use the shared loader.

## What Is Done

- Combined local branch created and renamed to `presentation`.
- Conflict-safe manifest created in `COMBINE_MANIFEST.md`.
- Root model definitions added in `src/models.py`.
- Root dependency file added in `requirements.txt`.
- Folder-aware FER2013 loader added in `src/data_loader.py`.
- HOG artifact trainer added in `src/train_hog_artifacts.py`.
- Demo app updated to load real CNN checkpoints and HOG artifacts when available.

## What Is Left

- Clean up Streamlit deprecation warnings for `use_container_width`.
- Final benchmark table for HOG, SimpleCNN, and LiteCNN.
- Comparison summary against the proposal targets and CLCM baseline.
- Optional cleanup of duplicate or branch-specific files if the repo needs a final tidy pass.

## Key Evidence Files

- Proposal: EE4016_Project_Proposal.md
- Integration contract: Integration_Architecture_Spec.md
- Demo app: demo_app.py
- Data pipeline: src/data_loader.py
- Baseline script: src/baselines.py
- Training scripts: src/train_simplecnn.py, src/train_litecnn.py, src/train_clcm.py, src/train_models.py
- Added root model definitions: src/models.py
- Added root dependencies: requirements.txt
- HOG artifact trainer: src/train_hog_artifacts.py
