from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import pickle
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

try:
    import torch
    from src.models import CLCM, LiteCNN, SimpleCNN

    TORCH_MODELS_AVAILABLE = True
except ImportError:
    TORCH_MODELS_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

APP_DIR = Path(__file__).resolve().parent


@dataclass
class PredictionResult:
    class_id: int
    label: str
    confidence: float
    probs: List[float]
    latency_ms: float
    model_name: str


class BaseAdapter:
    model_name: str
    model_family: str
    class_order: List[str]

    def predict(self, face_tensor: np.ndarray) -> PredictionResult:
        raise NotImplementedError


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def build_feature_vector(face_tensor: np.ndarray) -> np.ndarray:
    flat = face_tensor.flatten()
    mean_val = np.array([float(flat.mean())])
    std_val = np.array([float(flat.std())])
    min_val = np.array([float(flat.min())])
    max_val = np.array([float(flat.max())])

    grad_y, grad_x = np.gradient(face_tensor)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = (np.arctan2(grad_y, grad_x) + np.pi) / (2 * np.pi)

    hist_bins = 8
    hist = np.zeros(hist_bins, dtype=np.float32)
    bin_ids = np.minimum((grad_angle * hist_bins).astype(int), hist_bins - 1)
    for bin_index in range(hist_bins):
        hist[bin_index] = float(grad_mag[bin_ids == bin_index].sum())

    feature = np.concatenate([mean_val, std_val, min_val, max_val, hist])
    norm = np.linalg.norm(feature)
    return feature if norm == 0 else feature / norm


class HOGSVMAdapter(BaseAdapter):
    def __init__(self) -> None:
        self.model_name = "HOG+SVM (integration placeholder)"
        self.model_family = "hog_svm"
        self.class_order = EMOTIONS
        rng = np.random.default_rng(7)
        self.weights = rng.normal(0, 0.7, size=(len(EMOTIONS), 12)).astype(np.float32)
        self.bias = rng.normal(0, 0.2, size=(len(EMOTIONS),)).astype(np.float32)

    def predict(self, face_tensor: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        features = build_feature_vector(face_tensor)
        logits = self.weights @ features + self.bias
        probs = softmax(logits)
        class_id = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            class_id=class_id,
            label=self.class_order[class_id],
            confidence=float(probs[class_id]),
            probs=[float(x) for x in probs],
            latency_ms=float(latency_ms),
            model_name=self.model_name,
        )


def extract_hog_descriptor(face_tensor: np.ndarray, hog_config: Dict[str, Any]) -> np.ndarray:
    descriptor = hog(
        face_tensor,
        orientations=int(hog_config.get("orientations", 9)),
        pixels_per_cell=tuple(hog_config.get("pixels_per_cell", [8, 8])),
        cells_per_block=tuple(hog_config.get("cells_per_block", [2, 2])),
        visualize=False,
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


class HOGArtifactAdapter(BaseAdapter):
    def __init__(self, artifact_path: Path) -> None:
        self.model_name = "HOG+SVM (artifact)"
        self.model_family = "hog_svm"
        self.class_order = EMOTIONS
        self._artifact_path = artifact_path
        self._clf = None
        self._scaler = None
        self._hog_config: Dict[str, Any] = {
            "orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
        }
        self._load_error: Optional[str] = None

    def _lazy_load(self) -> None:
        if self._clf is not None or self._load_error is not None:
            return

        if not self._artifact_path.exists():
            self._load_error = f"HOG artifact not found: {self._artifact_path}"
            return

        try:
            with self._artifact_path.open("rb") as f:
                payload = pickle.load(f)
            self._clf = payload["classifier"]
            self._scaler = payload.get("scaler")
            self._hog_config = payload.get("hog_config", self._hog_config)
            self.class_order = payload.get("class_order", self.class_order)
        except Exception as exc:
            self._load_error = str(exc)

    def predict(self, face_tensor: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        self._lazy_load()
        if self._clf is None:
            error_detail = self._load_error or "unknown HOG adapter load error"
            raise RuntimeError(f"{self.model_name} unavailable: {error_detail}")

        features = extract_hog_descriptor(face_tensor, self._hog_config).reshape(1, -1)
        if self._scaler is not None:
            features = self._scaler.transform(features)

        decision = self._clf.decision_function(features)
        decision = np.asarray(decision, dtype=np.float32)
        if decision.ndim == 1:
            decision = decision.reshape(1, -1)

        probs = softmax(decision[0])
        class_id = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            class_id=class_id,
            label=self.class_order[class_id],
            confidence=float(probs[class_id]),
            probs=[float(x) for x in probs],
            latency_ms=float(latency_ms),
            model_name=self.model_name,
        )


class CNNStyleAdapter(BaseAdapter):
    def __init__(self, model_name: str, seed: int, scale: float) -> None:
        self.model_name = model_name
        self.model_family = "cnn"
        self.class_order = EMOTIONS
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, scale, size=(len(EMOTIONS), 12)).astype(np.float32)
        self.bias = rng.normal(0, 0.15, size=(len(EMOTIONS),)).astype(np.float32)

    def predict(self, face_tensor: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        features = build_feature_vector(face_tensor)
        logits = self.weights @ features + self.bias
        probs = softmax(logits)
        class_id = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            class_id=class_id,
            label=self.class_order[class_id],
            confidence=float(probs[class_id]),
            probs=[float(x) for x in probs],
            latency_ms=float(latency_ms),
            model_name=self.model_name,
        )


class TorchCheckpointAdapter(BaseAdapter):
    def __init__(
        self,
        model_name: str,
        model_family: str,
        model_cls: Type,
        checkpoint_path: Path,
    ) -> None:
        self.model_name = model_name
        self.model_family = model_family
        self.class_order = EMOTIONS
        self._model_cls = model_cls
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._load_error: Optional[str] = None

    def _lazy_load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        if not TORCH_MODELS_AVAILABLE:
            self._load_error = "PyTorch or src.models import failed"
            return

        if not self._checkpoint_path.exists():
            self._load_error = f"Checkpoint not found: {self._checkpoint_path}"
            return

        try:
            model = self._model_cls(num_classes=len(self.class_order))
            state = torch.load(str(self._checkpoint_path), map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._model = model
        except Exception as exc:
            self._load_error = str(exc)

    def predict(self, face_tensor: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        self._lazy_load()
        if self._model is None:
            error_detail = self._load_error or "unknown adapter load error"
            raise RuntimeError(f"{self.model_name} unavailable: {error_detail}")

        input_tensor = torch.from_numpy(face_tensor).float().unsqueeze(0).unsqueeze(0)
        # Match training-time normalization from data_loader.py
        input_tensor = (input_tensor - 0.5) / 0.5

        with torch.no_grad():
            logits = self._model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_id = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            class_id=class_id,
            label=self.class_order[class_id],
            confidence=float(probs[class_id]),
            probs=[float(x) for x in probs],
            latency_ms=float(latency_ms),
            model_name=self.model_name,
        )


@st.cache_resource
def get_adapter_registry() -> Dict[str, BaseAdapter]:
    registry: Dict[str, BaseAdapter] = {}

    hog_artifact = APP_DIR / "saved_models" / "hog_svm_artifact.pkl"
    if hog_artifact.exists():
        registry["HOG+SVM"] = HOGArtifactAdapter(artifact_path=hog_artifact)
    else:
        registry["HOG+SVM"] = HOGSVMAdapter()

    simple_ckpt = APP_DIR / "saved_models" / "SimpleCNN_best.pth"
    lite_ckpt = APP_DIR / "saved_models" / "LiteCNN_best.pth"
    clcm_ckpt = APP_DIR / "saved_models" / "clcm_best_weights.pth"

    if TORCH_MODELS_AVAILABLE and simple_ckpt.exists():
        registry["SimpleCNN"] = TorchCheckpointAdapter(
            model_name="SimpleCNN (checkpoint)",
            model_family="simplecnn",
            model_cls=SimpleCNN,
            checkpoint_path=simple_ckpt,
        )
    else:
        registry["SimpleCNN"] = CNNStyleAdapter(
            "SimpleCNN (fallback placeholder)", seed=23, scale=0.65
        )

    if TORCH_MODELS_AVAILABLE and lite_ckpt.exists():
        registry["LiteCNN"] = TorchCheckpointAdapter(
            model_name="LiteCNN (checkpoint)",
            model_family="litecnn",
            model_cls=LiteCNN,
            checkpoint_path=lite_ckpt,
        )
    else:
        registry["LiteCNN"] = CNNStyleAdapter(
            "LiteCNN (fallback placeholder)", seed=47, scale=0.60
        )

    if TORCH_MODELS_AVAILABLE and clcm_ckpt.exists():
        registry["CLCM"] = TorchCheckpointAdapter(
            model_name="CLCM (checkpoint)",
            model_family="clcm",
            model_cls=CLCM,
            checkpoint_path=clcm_ckpt,
        )
    else:
        registry["CLCM"] = CNNStyleAdapter(
            "CLCM (fallback placeholder)", seed=61, scale=0.58
        )

    return registry


def preprocess_image_to_face_tensor(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L")
    resized = grayscale.resize((48, 48))
    face_tensor = np.asarray(resized, dtype=np.float32) / 255.0
    return face_tensor


def l2_change_score(current_face: np.ndarray, previous_face: Optional[np.ndarray]) -> float:
    if previous_face is None:
        return 1.0
    diff = current_face - previous_face
    return float(np.sqrt(np.mean(diff ** 2)))


def should_infer(
    mode: str,
    now_ts: float,
    last_infer_ts: Optional[float],
    interval_seconds: float,
    change_score: float,
    change_threshold: float,
) -> Tuple[bool, str]:
    if last_infer_ts is None:
        return True, "First frame"

    timed_ok = (now_ts - last_infer_ts) >= interval_seconds
    changed_ok = change_score >= change_threshold

    if mode == "Timed":
        return timed_ok, "Timed trigger" if timed_ok else "Timed skip"
    if mode == "Change-triggered":
        return changed_ok, "Change trigger" if changed_ok else "Change skip"

    return False, "Invalid sampling mode"


def init_session_state() -> None:
    defaults = {
        "last_infer_ts": None,
        "last_face_tensor": None,
        "last_prediction": None,
        "infer_calls_count": 0,
        "skipped_frames_count": 0,
        "total_frames_count": 0,
        "last_decision_reason": "-",
        "last_change_score": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_live_runtime_state() -> Dict[str, Any]:
    return {
        "lock": Lock(),
        "last_infer_ts": None,
        "last_face_tensor": None,
        "last_prediction": None,
        "infer_calls_count": 0,
        "skipped_frames_count": 0,
        "total_frames_count": 0,
        "last_decision_reason": "-",
        "last_change_score": 0.0,
        "selected_model_key": "LiteCNN",
        "sampling_mode": "Timed",
        "interval_seconds": 1.0,
        "change_threshold": 0.050,
    }


def reset_live_runtime_state() -> None:
    runtime = get_live_runtime_state()
    lock = runtime["lock"]

    with lock:
        runtime["last_infer_ts"] = None
        runtime["last_face_tensor"] = None
        runtime["last_prediction"] = None
        runtime["infer_calls_count"] = 0
        runtime["skipped_frames_count"] = 0
        runtime["total_frames_count"] = 0
        runtime["last_decision_reason"] = "-"
        runtime["last_change_score"] = 0.0


def update_live_runtime_config(
    selected_model_key: str,
    sampling_mode: str,
    interval_seconds: float,
    change_threshold: float,
) -> None:
    runtime = get_live_runtime_state()
    lock = runtime["lock"]

    with lock:
        runtime["selected_model_key"] = selected_model_key
        runtime["sampling_mode"] = sampling_mode
        runtime["interval_seconds"] = interval_seconds
        runtime["change_threshold"] = change_threshold


def get_live_runtime_snapshot() -> Dict[str, Any]:
    runtime = get_live_runtime_state()
    lock = runtime["lock"]

    with lock:
        return {
            "last_prediction": runtime["last_prediction"],
            "infer_calls_count": runtime["infer_calls_count"],
            "skipped_frames_count": runtime["skipped_frames_count"],
            "total_frames_count": runtime["total_frames_count"],
            "last_decision_reason": runtime["last_decision_reason"],
            "last_change_score": runtime["last_change_score"],
        }


def render_prediction(prediction: PredictionResult) -> None:
    st.subheader("Prediction")
    st.write(f"Model: **{prediction.model_name}**")
    st.write(f"Emotion: **{prediction.label}**")
    st.write(f"Confidence: **{prediction.confidence:.3f}**")
    st.write(f"Latency: **{prediction.latency_ms:.2f} ms**")

    st.caption("Class probabilities (canonical class order)")
    prob_map = {emotion: score for emotion, score in zip(EMOTIONS, prediction.probs)}
    st.bar_chart(prob_map)


def render_sampling_stats() -> None:
    render_sampling_stats_values(
        total=st.session_state.total_frames_count,
        infer_calls=st.session_state.infer_calls_count,
        skips=st.session_state.skipped_frames_count,
        last_reason=st.session_state.last_decision_reason,
        last_change=st.session_state.last_change_score,
    )


def render_sampling_stats_values(
    total: int,
    infer_calls: int,
    skips: int,
    last_reason: str,
    last_change: float,
) -> None:
    reduction = (skips / total) * 100 if total else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", total)
    col2.metric("Inferences", infer_calls)
    col3.metric("Skipped", skips)
    col4.metric("Reduction", f"{reduction:.1f}%")

    st.caption(f"Last decision: {last_reason} | Last L2 change: {last_change:.4f}")


def make_live_video_callback():
    def video_frame_callback(frame):
        runtime = get_live_runtime_state()
        lock = runtime["lock"]

        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)
        face_tensor = preprocess_image_to_face_tensor(pil_image)

        with lock:
            selected_model_key = runtime["selected_model_key"]
            sampling_mode = runtime["sampling_mode"]
            interval_seconds = runtime["interval_seconds"]
            change_threshold = runtime["change_threshold"]

            previous_face = runtime["last_face_tensor"]
            change_score = l2_change_score(face_tensor, previous_face)
            now_ts = time.time()
            should_run, reason = should_infer(
                mode=sampling_mode,
                now_ts=now_ts,
                last_infer_ts=runtime["last_infer_ts"],
                interval_seconds=interval_seconds,
                change_score=change_score,
                change_threshold=change_threshold,
            )

            runtime["total_frames_count"] += 1
            runtime["last_face_tensor"] = face_tensor
            runtime["last_change_score"] = change_score
            runtime["last_decision_reason"] = reason

            if should_run:
                adapter = get_adapter_registry()[selected_model_key]
                prediction = adapter.predict(face_tensor)
                runtime["last_prediction"] = prediction
                runtime["last_infer_ts"] = now_ts
                runtime["infer_calls_count"] += 1
            else:
                runtime["skipped_frames_count"] += 1

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")

    return video_frame_callback


def run_inference_flow(image: Image.Image, selected_model_key: str, mode: str, interval: float, threshold: float) -> None:
    adapters = get_adapter_registry()
    adapter = adapters[selected_model_key]

    face_tensor = preprocess_image_to_face_tensor(image)
    now_ts = time.time()
    change_score = l2_change_score(face_tensor, st.session_state.last_face_tensor)

    should_run, reason = should_infer(
        mode=mode,
        now_ts=now_ts,
        last_infer_ts=st.session_state.last_infer_ts,
        interval_seconds=interval,
        change_score=change_score,
        change_threshold=threshold,
    )

    st.session_state.total_frames_count += 1
    st.session_state.last_change_score = change_score
    st.session_state.last_decision_reason = reason
    st.session_state.last_face_tensor = face_tensor

    if should_run:
        prediction = adapter.predict(face_tensor)
        st.session_state.last_prediction = prediction
        st.session_state.last_infer_ts = now_ts
        st.session_state.infer_calls_count += 1
    else:
        st.session_state.skipped_frames_count += 1

    if st.session_state.last_prediction is not None:
        render_prediction(st.session_state.last_prediction)

    render_sampling_stats()


def run_single_image_inference(image: Image.Image, selected_model_key: str) -> None:
    adapters = get_adapter_registry()
    adapter = adapters[selected_model_key]
    face_tensor = preprocess_image_to_face_tensor(image)
    prediction = adapter.predict(face_tensor)

    st.session_state.last_prediction = prediction
    st.session_state.last_face_tensor = face_tensor
    st.session_state.last_decision_reason = "Manual image inference"
    st.session_state.last_change_score = 0.0
    st.session_state.total_frames_count += 1
    st.session_state.infer_calls_count += 1

    render_prediction(prediction)
    render_sampling_stats()


def main() -> None:
    st.set_page_config(page_title="MoodMirror FER Demo", layout="centered")
    st.title("MoodMirror: Facial Expression Recognition Demo")
    st.write("Integration-ready UI for HOG+SVM, SimpleCNN, LiteCNN, and CLCM adapters.")

    init_session_state()

    with st.sidebar:
        st.header("Settings")
        model_options = list(get_adapter_registry().keys())
        default_index = model_options.index("LiteCNN") if "LiteCNN" in model_options else 0
        selected_model = st.selectbox("Model", model_options, index=default_index)
        input_source = st.radio("Input source", ["Image Upload", "Webcam Snapshot", "Live Video"], index=0)

        sampling_mode = "Timed"
        interval_seconds = 1.0
        change_threshold = 0.050
        if input_source in ["Webcam Snapshot", "Live Video"]:
            sampling_mode = st.selectbox("Sampling mode", ["Timed", "Change-triggered"], index=0)
            interval_seconds = st.slider("Interval (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
            change_threshold = st.slider("Change threshold (L2)", min_value=0.001, max_value=0.300, value=0.050, step=0.001)

        if st.button("Reset sampling counters"):
            st.session_state.last_infer_ts = None
            st.session_state.last_face_tensor = None
            st.session_state.last_prediction = None
            st.session_state.infer_calls_count = 0
            st.session_state.skipped_frames_count = 0
            st.session_state.total_frames_count = 0
            st.session_state.last_decision_reason = "-"
            st.session_state.last_change_score = 0.0
            reset_live_runtime_state()

    st.info(
        "SimpleCNN/LiteCNN/CLCM now use checkpoint adapters when available in saved_models/. "
        "If checkpoint loading fails, the app automatically falls back to placeholder adapters."
    )

    st.caption(
        "Current app modes: Normal image prediction, webcam snapshot with Timed or Motion-triggered sampling, "
        "and Live Video using the same smart sampling rules."
    )

    if input_source == "Image Upload":
        uploaded = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Input image", use_container_width=True)
            run_single_image_inference(image=image, selected_model_key=selected_model)
    
    if input_source == "Image Upload":
        st.caption("You can switch model in the sidebar and reuse the same uploaded image without retaking it.")
    elif input_source == "Webcam Snapshot":
        captured = st.camera_input("Capture webcam snapshot")
        if captured is not None:
            image = Image.open(captured)
            st.image(image, caption="Captured frame", use_container_width=True)
            run_inference_flow(
                image=image,
                selected_model_key=selected_model,
                mode=sampling_mode,
                interval=interval_seconds,
                threshold=change_threshold,
            )
    else:
        if not WEBRTC_AVAILABLE:
            st.error(
                "Live Video requires streamlit-webrtc and av. Install with: pip install streamlit-webrtc av"
            )
        else:
            update_live_runtime_config(
                selected_model_key=selected_model,
                sampling_mode=sampling_mode,
                interval_seconds=interval_seconds,
                change_threshold=change_threshold,
            )

            st.write("Start the webcam stream below. Smart sampling runs on each incoming frame.")
            webrtc_streamer(
                key="moodmirror-live-video",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=make_live_video_callback(),
                async_processing=True,
            )

            live_snapshot = get_live_runtime_snapshot()
            live_prediction = live_snapshot["last_prediction"]
            if live_prediction is not None:
                render_prediction(live_prediction)

            render_sampling_stats_values(
                total=int(live_snapshot["total_frames_count"]),
                infer_calls=int(live_snapshot["infer_calls_count"]),
                skips=int(live_snapshot["skipped_frames_count"]),
                last_reason=str(live_snapshot["last_decision_reason"]),
                last_change=float(live_snapshot["last_change_score"]),
            )


if __name__ == "__main__":
    main()
