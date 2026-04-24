# TECHNICAL REPORT

## MoodMirror: Lightweight Facial Expression Recognition with Convolutional Neural Networks

### Group 10, EE4016

- KAPYA Zachariah Muya (Leader)
- WANG Xinan
- Cheng Wing Yan
- YIP Chung Hon
- Sit Yan Tung

Date: April 24, 2026
Version: 1.0 (Draft for final LaTeX formatting)

---

## Revision History

| Version | Date | Comments |
|---|---|---|
| 0.1 | 2026-04-12 | Presentation branch consolidation and integration checkpoints verified |
| 0.2 | 2026-04-24 | Verified benchmark rerun on current artifacts and complete technical report draft |
| 1.0 | 2026-04-24 | Full report text compiled in Markdown for LaTeX visual editor transfer |

---

## Abstract

This report presents MoodMirror, a lightweight facial expression recognition (FER) system for seven-class emotion classification (angry, disgust, fear, happy, sad, surprise, neutral) on FER2013. The project compares a classical machine learning baseline (HOG+SVM), a deep learning baseline (SimpleCNN), and a proposed lightweight architecture (LiteCNN), with a replication reference model (CLCM-style lightweight CNN). The system was developed under reproducibility and integration constraints: common data loader, shared inference contract, and a unified Streamlit demonstration interface.

On the current repository artifacts, verified test results are 43.23% (HOG+SVM), 60.39% (CLCM), and 65.88% (LiteCNN). LiteCNN achieves 900,903 parameters and outperforms CLCM by +5.49 percentage points on test accuracy while remaining below the project parameter budget (<1.5M). The deployed demo supports image upload, webcam snapshot, and live video with intelligent frame sampling to reduce unnecessary inference calls. Remaining gaps to proposal targets are the final >=70% verified LiteCNN checkpoint and final tabulated latency measurements under fixed hardware conditions.

---

## 1. Introduction

### 1.1 Problem Context

Facial expression recognition is a challenging visual classification task due to pose variations, lighting changes, occlusions, subtle inter-class differences, and class imbalance. While large deep networks can improve recognition performance, they are often computationally expensive and less practical for CPU-constrained, real-time applications.

MoodMirror addresses this gap by focusing on lightweight FER models and integration-ready deployment. The project emphasizes two dimensions simultaneously:

- Predictive quality (accuracy, macro F1, per-class reliability)
- Deployment practicality (parameter count, inference latency, simplified integration)

### 1.2 Project Scope

The work includes:

- Dataset pipeline for FER2013 folder format
- Baseline HOG+SVM model
- Deep CNN baselines and lightweight model development
- CLCM-style replication path for paper comparison reference
- Ablation-oriented training scripts and configuration points
- Integrated Streamlit app with unified model adapters
- Intelligent webcam sampling for reduced inference load

### 1.3 Core Contributions

- Consolidated multi-branch project integration into one runnable presentation branch
- Standardized model inference interface across HOG and CNN families
- Artifact-based inference in demo (real checkpoints and exported HOG model)
- Verified benchmark rerun on current saved artifacts
- Objective-level status mapping from proposal to achieved outcomes

---

## 2. Objectives and Success Criteria

The proposal defined six objectives:

- O1: Build end-to-end FER pipeline (data, training, evaluation, demo)
- O2: Establish baselines (HOG+SVM and SimpleCNN)
- O3: Develop LiteCNN (<1.5M params, <10ms CPU inference, >=70% accuracy)
- O4: Quantify regularization and optimization via ablations
- O5: Deliver interactive demo (image upload + intelligent webcam sampling)
- O6: Beat CLCM lightweight baseline

This report evaluates each objective using repository evidence and verified current artifacts.

---

## 3. Repository Consolidation and Integration Evidence

### 3.1 Branch and Source Consolidation

The final integration was performed in branch `presentation`, consolidating work from:

- `annie_m2_code`
- `jasmine_code`
- `master`
- `muyas_code`

Remote project branches additionally include `james_m4` and `melody_code_m5`, with integration-level documentation and outputs represented through consolidated root files and report notes.

### 3.2 Conflict-Safe Merge Strategy

A manifest-based merge strategy was used:

- Non-conflicting paths promoted to root
- Conflicting paths preserved and resolved with branch-safe handling

Conflicting files included `.gitignore`, `README.md`, `requirements.txt`, and `src/models.py`.

### 3.3 Integration Artifacts

Core integrated files include:

- Demo app: `demo_app.py`
- Shared models: `src/models.py`
- Shared data pipeline: `src/data_loader.py`
- Baseline pipeline: `src/baselines.py`
- CNN training: `src/train_simplecnn.py`, `src/train_litecnn.py`, `src/train_models.py`
- CLCM training: `src/train_clcm.py`
- HOG artifact export path: `src/train_hog_artifacts.py`
- Saved artifacts: `saved_models/*.pth`, `saved_models/hog_svm_artifact.pkl`

---

## 4. Dataset and Preprocessing

### 4.1 Dataset

FER2013 is used with the folder layout:

- `data/fer2013/train/<emotion>`
- `data/fer2013/test/<emotion>`

Seven emotion classes are used consistently across all model families:

- angry, disgust, fear, happy, sad, surprise, neutral

### 4.2 Data Loading Contract

The custom `FER2013Dataset` supports:

- CSV mode (legacy FER2013 format)
- Folder mode (current integrated usage)

Split behavior in folder mode:

- Training: class-wise train subset from `train`
- Validation: class-wise holdout (from `train`, val fraction default 0.1)
- Test: full `test` folder as private test split

### 4.3 Data Transforms

Training transforms:

- Random horizontal flip
- Random rotation
- Random affine translation
- Mild brightness/contrast jitter
- Tensor conversion and normalization (mean=0.5, std=0.5)

Validation/test transforms:

- Tensor conversion
- Same normalization

---

## 5. Methods and Model Architectures

### 5.1 Baseline 1: HOG+SVM

The classical baseline pipeline:

- HOG descriptor extraction (orientations=9, 8x8 pixels/cell, 2x2 cells/block)
- Linear SVM classifier (`LinearSVC`)
- Test metrics: accuracy, macro F1, confusion matrix

This baseline provides a non-deep-learning reference point.

### 5.2 Baseline 2: SimpleCNN

SimpleCNN architecture in integrated root:

- 4 convolutional blocks with batch normalization
- Max pooling and dropout regularization
- Global average pooling and linear classifier

Parameter count (trainable): 390,599

### 5.3 Proposed Model: LiteCNN

LiteCNN architecture:

- Initial convolution stem
- Residual blocks built from depthwise separable convolutions
- Progressive channel expansion (32 -> 48 -> 96 -> 192 -> 384)
- Global average pooling + dropout + linear classifier

Parameter count (trainable): 900,903

### 5.4 Replication Reference: CLCM

CLCM-like model:

- Depthwise separable convolution pipeline
- Lightweight classifier head

Parameter count (trainable): 117,383

The CLCM path is used as the proposal comparison baseline.

---

## 6. Training and Evaluation Protocol

### 6.1 Training Configuration

Common training scripts include:

- Cross-entropy loss
- Adam optimizer with weight decay
- Scheduler usage (CosineAnnealingLR in SimpleCNN/LiteCNN scripts; ReduceLROnPlateau in CLCM script)
- Checkpoint save on best validation accuracy

Representative defaults:

- Epochs: 50
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 1e-4

### 6.2 Metrics

Primary metrics:

- Accuracy
- Macro F1-score

Additional metrics:

- Parameter count
- Inference latency (tracked in demo output and measured in current verification script)

### 6.3 Evidence Policy Used in This Report

To avoid overstating outcomes, results are split into:

- Verified current artifacts: rerun directly from `saved_models` on current test set
- Historical notes: values documented in prior branch notes/READMEs but not currently reproducible from the same checkpoint set

---

## 7. Integrated Demo System

### 7.1 Unified Adapter Contract

The Streamlit app uses a unified adapter interface for all models. Required prediction output fields are:

- `class_id`
- `label`
- `confidence`
- `probs`
- `latency_ms`
- `model_name`

This allows UI logic to remain model-agnostic.

### 7.2 Inference Paths

Current app supports:

- Image upload
- Webcam snapshot
- Live video stream (with streamlit-webrtc)

Model loading behavior:

- If checkpoint/artifact exists and loads, inference uses real model
- If loading fails, fallback placeholder adapter is used

### 7.3 Intelligent Sampling

For webcam and live video modes, the app supports:

- Timed sampling
- Change-triggered sampling using L2 frame change score

Displayed runtime KPIs:

- Frames processed
- Inference calls
- Skipped frames
- Reduction percentage

This is a practical edge-oriented engineering contribution aligned with CPU deployment constraints.

---

## 8. Results

## 8.1 Verified Current Benchmark (April 24, 2026)

All values below were re-evaluated from current artifacts in `saved_models` on the current FER2013 test folder.

| Model | Test Accuracy (%) | Macro F1 | Params | Avg Latency (ms/sample, CPU run) | Notes |
|---|---:|---:|---:|---:|---|
| HOG+SVM (artifact) | 43.23 | 0.3803 | N/A | N/A | Evaluated from `hog_svm_artifact.pkl` |
| SimpleCNN (checkpoint) | 25.65 | 0.0840 | 390,599 | 1.05 | Checkpoint loads but performs far below historical values |
| LiteCNN (checkpoint) | 65.88 | 0.6340 | 900,903 | 9.48 | Best current verified performer |
| CLCM (checkpoint) | 60.39 | 0.5268 | 117,383 | 2.15 | Replication baseline reference |

### 8.2 Historical Recorded Results (from Project Notes)

Historical values from prior runs include:

- HOG+SVM: 43.23% test, macro F1 0.3803
- SimpleCNN: 62.44% val, 63.03% test
- LiteCNN: 65.73% val, 66.09% test
- Additional expected/target-oriented run logs show higher values in README examples, but those are not treated as current verified results.

### 8.3 Key Comparative Findings

- LiteCNN vs CLCM (verified current):
  - Test accuracy gain: +5.49 percentage points (65.88 - 60.39)
  - Higher macro F1 on current checkpoint set
- LiteCNN parameter budget objective:
  - Met (<1.5M), with 900,903 parameters
- HOG+SVM remains a useful classical anchor but significantly below CNN performance

### 8.4 Result Integrity Note

SimpleCNN currently shows a reproducibility mismatch between historical reported performance and present checkpoint behavior. This report treats the current checkpoint rerun as authoritative for final benchmarking unless checkpoint lineage is reconciled before final submission.

---

## 9. Ablation and Optimization Discussion

The repository includes ablation-oriented design hooks and configuration targets for:

- Data augmentation on/off
- Dropout variations
- Weight decay strength
- Optimizer/scheduler choices
- Architecture comparison (SimpleCNN vs LiteCNN)

Current status:

- Ablation framework and script-level knobs are present
- Partial historical outcomes are documented
- A fully frozen, publication-style ablation table with controlled reruns should be finalized before submission lock

Recommended final ablation table fields:

- Experiment ID
- Model
- Changed factor
- Validation accuracy
- Test accuracy
- Macro F1
- Parameter count
- Comments

---

## 10. Objective-by-Objective Achievement Assessment

| Objective | Target | Current Status | Evidence |
|---|---|---|---|
| O1 End-to-end FER pipeline | Data + training + eval + demo | Achieved | Integrated scripts and runnable Streamlit demo |
| O2 Baselines | HOG+SVM + SimpleCNN | Achieved (with caveat on SimpleCNN checkpoint quality) | Baseline scripts and saved artifacts |
| O3 LiteCNN target | <1.5M params, <10ms CPU, >=70% acc | Partially achieved | Params met (900,903), latency near bound (9.48 ms/sample in current run), accuracy currently 65.88% |
| O4 Ablation quantification | E1-E5 analysis | In progress / partial | Script and configuration support present; final controlled table pending |
| O5 Interactive demo | Upload + webcam + smart sampling | Achieved | `demo_app.py` supports all required modes |
| O6 Beat CLCM | LiteCNN > CLCM on performance | Achieved (current checkpoints) | 65.88% vs 60.39% test accuracy |

---

## 11. Limitations and Risks

### 11.1 Checkpoint Lineage Consistency

The largest risk is mismatch between historical reported values and currently reproducible checkpoint outcomes, especially for SimpleCNN.

### 11.2 Final Latency Reporting Standard

Latency values should be re-measured using a fixed protocol:

- same hardware
- same batch size (prefer batch=1 for deployment realism)
- warm-up and repeated trials
- average and variance reporting

### 11.3 Class-Level Error Analysis Depth

Confusion-matrix and per-class precision/recall should be included in the final polished report figures to strengthen interpretation.

---

## 12. Conclusion

MoodMirror successfully delivered an integrated lightweight FER system from multi-branch team contributions into a single runnable pipeline. The project demonstrates clear engineering maturity in unified data handling, model adapter architecture, artifact-based inference, and deployment-oriented webcam sampling.

On verified current artifacts, LiteCNN is the strongest model and outperforms the replicated CLCM baseline while meeting parameter-budget constraints. The final technical gap to the original proposal is achieving and freezing a >=70% verified LiteCNN checkpoint and completing final standardized latency and ablation tables for publication-quality reporting.

Overall, the project meets its core integration and comparison goals and provides a strong foundation for final optimization and polished submission.

---

## 13. Final Submission Checklist Mapping

Required by course:

- Final technical report (25+ pages): This Markdown draft provides complete content for LaTeX transfer
- PowerPoint presentation: to include benchmark and architecture summary tables
- Python source code: included in repository
- Demo video (3-4 minutes): demo app supports all required interaction modes
- Appendix A individual contributions: included below
- Single zip packaging: pending final assembly

---

## References

1. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.
2. Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436-444, 2015.
3. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *CVPR*, 2005.
4. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273-297, 1995.
5. A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in *NIPS*, 2012.
6. A. G. Howard et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv:1704.04861, 2017.
7. M. C. Gursesli et al., "Facial Emotion Recognition (FER) Through Custom Lightweight CNN Model: Performance Evaluation in Public Datasets," *IEEE Access*, 2024, doi:10.1109/ACCESS.2024.3380847.
8. I. J. Goodfellow et al., "Challenges in representation learning: A report on three machine learning contests," *Neural Networks*, vol. 64, pp. 59-63, 2015.
9. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *CVPR*, 2016.
10. F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in *CVPR*, 2017.
11. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *ICML*, 2015.
12. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *ICLR*, 2015.
13. T.-Y. Lin et al., "Feature pyramid networks for object detection," in *CVPR*, 2017.
14. J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *CVPR*, 2009.
15. P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," in *CVPR*, 2001.

---

## Appendix A: Individual Contributions of the Group Project

### A.1 Contribution Summary by Member

| Member | Planned Role (Proposal) | Implemented Contributions | Current Evidence |
|---|---|---|---|
| M1 (Integration Lead) | App integration, intelligent sampling, coordination | Consolidated branch integration, unified demo adapters, live/webcam/image modes, runtime sampling metrics | `demo_app.py`, `Integration_Architecture_Spec.md`, `COMBINE_MANIFEST.md` |
| M2 | Data pipeline, HOG+SVM baseline, CLCM replication support | Shared FER loader design, HOG baseline structure, baseline evaluation workflow | `src/data_loader.py`, `src/baselines.py`, `README_annie_m2.md` |
| M3 | SimpleCNN and LiteCNN implementation | Implemented CNN models and associated training scripts | `src/models.py`, `src/train_simplecnn.py`, `src/train_litecnn.py`, `README_jasmine.md` |
| M4 | Ablation experiment design and execution | Hyperparameter/ablation knobs embedded in training scripts and configs path | `src/train_models.py`, `m4/configs/*`, experiment scripts in `m4/experiments/*` |
| M5 | Evaluation and analysis support | Metrics interpretation and reporting support reflected in integration notes and final benchmark reconciliation | report notes, consolidated benchmark analysis |

### A.2 Team-Level Collaboration Outcomes

- Common data contract enabled cross-member script compatibility
- Adapter contract decoupled UI from model internals
- Branch consolidation reduced merge risk near deadline
- Shared artifact folder enabled reproducible demo inference

### A.3 Remaining Individual/Joint Follow-Up Before Final Freeze

- Reconcile SimpleCNN checkpoint lineage (owner + integration verification)
- Lock final LiteCNN optimization run for >=70% target attempt
- Produce final ablation table with controlled rerun IDs
- Finalize report figures and per-class confusion matrix visuals

---

## Appendix B: Suggested Figure/Table Placeholders for LaTeX Version

For the final LaTeX layout, include:

- Figure B1: System architecture flowchart (input -> preprocessing -> adapter -> output)
- Figure B2: Demo UI screenshots (upload, webcam snapshot, live sampling)
- Figure B3: Confusion matrix (HOG+SVM)
- Figure B4: Confusion matrix (LiteCNN current best checkpoint)
- Table B1: Final benchmark table (verified values)
- Table B2: Objective mapping table (O1-O6)
- Table B3: Ablation summary (E1-E5)

