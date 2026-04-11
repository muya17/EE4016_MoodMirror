# MoodMirror Integration Architecture Spec

## 1) Purpose
This document defines the integration contract between model contributors and the Streamlit app.

Primary goal:
- Let M1 integrate all models consistently without changing UI logic per model.

Secondary goal:
- Ensure reproducible, comparable outputs across HOG+SVM, SimpleCNN, LiteCNN, and optional distilled LiteCNN.

---

## 2) Why This Structure
This architecture uses a single adapter contract so different model families (HOG+SVM and CNN variants) can be integrated through the same UI path without custom per-model app logic.

We chose this structure because it:
- keeps integration fast and low-risk near deadlines,
- preserves fair model comparison with consistent outputs,
- allows parallel team work (training and UI can progress independently),
- reduces merge conflicts by separating model internals from UI behavior.

---

## 3) System Overview
High-level pipeline:
1. Input source (image upload or webcam frame)
2. Face localization/crop
3. Shared normalization checks
4. Model-specific preprocessing
5. Model inference through adapter
6. Standardized output formatting
7. UI render (label, confidence, class scores, latency)
8. Optional intelligent sampling gate for webcam mode

Design rule:
- UI layer never depends on model internals.
- All models must pass through one adapter contract.

---

## 4) Model Integration Contract
Each model must provide an adapter that satisfies the same API.

## 4.1 Required metadata
- model_name: string
- model_family: one of [hog_svm, simplecnn, litecnn, litecnn_kd]
- class_order: list of 7 labels in exact output order
- expected_input: description of input requirements
- version_tag: semantic or checkpoint identifier

## 4.2 Required inference behavior
Input:
- face crop image from shared face stage

Output (required keys):
- class_id: int in [0..6]
- label: string
- confidence: float in [0,1]
- probs: list of 7 floats summing to approximately 1
- latency_ms: float

Failure behavior:
- Return structured error with message and cause category (preprocess, model_load, inference).
- Do not crash app loop.

## 4.3 Output schema (canonical)
{
  "class_id": 3,
  "label": "happy",
  "confidence": 0.88,
  "probs": [0.01, 0.00, 0.02, 0.88, 0.03, 0.04, 0.02],
  "latency_ms": 6.4,
  "model_name": "LiteCNN_v1"
}

---

## 5) Preprocessing Contract
Shared preprocessing before adapter:
- detect/select face region
- crop and align to face ROI
- convert to grayscale when required
- resize to target resolution (default 48x48)

Model-specific preprocessing (inside adapter):
- HOG+SVM: HOG descriptor extraction and shape flattening
- CNN-family: tensor conversion, normalization, channel formatting

Critical requirement:
- class order and preprocessing assumptions must be documented with each checkpoint.

---

## 6) Intelligent Sampling Contract (Webcam)
Applies to webcam mode only.

Sampling modes:
- Timed mode: run inference every N seconds
- Change-triggered mode: run inference when L2 difference exceeds threshold
- Hybrid mode (optional): trigger if either condition is met

Required UI controls:
- sampling_mode
- interval_seconds
- change_threshold

State needed in session:
- last_infer_timestamp
- last_face_crop (or feature snapshot)
- infer_calls_count
- skipped_frames_count

KPI shown in demo:
- total frames
- inferences run
- inference reduction percentage

---

## 7) Streamlit Architecture Guidelines
Mental model:
- Script reruns top-to-bottom on each interaction.

Rules:
- Keep heavy model loading cached.
- Keep only interaction/runtime variables in session state.
- Keep inference and preprocessing in pure functions where possible.
- Avoid UI code that mutates model internals directly.

Recommended app modules:
- ui_controls: widgets and settings
- input_handlers: upload/webcam read
- preprocess: shared face pipeline
- model_adapters: one adapter per model family
- sampling: timed/change gating logic
- renderers: prediction and metrics display

---

## 8) Team Handoff Requirements
## 8.1 Model-team deliverables for integration
Each model handoff must include:
- adapter implementation following the contract in Section 4
- class order declaration in exact output order
- preprocessing note for inference-time consistency
- checkpoint/model artifact with version tag

For comparison/report assets, model and evaluation contributors should provide:
- baseline/ablation summary tables
- confusion matrix and per-class metrics

## 8.2 Integration outputs to team
The integration flow should produce:
- stable app build using one shared model contract
- reproducible demo run path
- consistent UI outputs for model comparison
- demo-ready screenshots/video captures

---

## 9) Acceptance Criteria (Definition of Integration Done)
A model is considered integration-ready only if all pass:
1. Adapter returns canonical output schema.
2. Class order matches global class order exactly.
3. Inference runs from app without code changes in UI layer.
4. Prediction + confidence + latency are rendered correctly.
5. Webcam sampling works with selected model.
6. Failure paths are handled without app crash.

Project integration is done when:
- HOG+SVM, SimpleCNN, and LiteCNN all run via same UI contract.
- Best final model is selected and frozen for demo.
- Intelligent sampling metrics are visible in demo.

---

## 10) Immediate Next Actions (No Code Redesign)
1. Share this spec with M2/M3/M4/M5.
2. Collect adapter readiness status from M2 and M3.
3. Lock final class order and output schema in team chat.
4. Start integration using current demo app skeleton.
5. Freeze first end-to-end demo candidate build.
