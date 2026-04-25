import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from data_loader import FER2013Dataset

EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


@dataclass
class Config:
    preprocess: str
    orientations: int
    pixels_per_cell: Tuple[int, int]
    cells_per_block: Tuple[int, int]
    classifier: str
    c_value: float


@dataclass
class Result:
    config: Config
    val_accuracy: float
    val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    confusion_matrix: List[List[int]]


def preprocess_image(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    if mode == "equalize":
        return cv2.equalizeHist(img)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    raise ValueError(f"Unsupported preprocess mode: {mode}")


def extract_hog_features(
    dataset: FER2013Dataset,
    preprocess_mode: str,
    orientations: int,
    pixels_per_cell: Tuple[int, int],
    cells_per_block: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels = np.asarray(dataset.emotions, dtype=np.int64)

    for image in dataset.images:
        img_u8 = np.asarray(image, dtype=np.uint8)
        img_u8 = preprocess_image(img_u8, preprocess_mode)
        descriptor = hog(
            img_u8,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )
        features.append(descriptor.astype(np.float32))

    return np.asarray(features, dtype=np.float32), labels


def maybe_subsample(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return x, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), size=max_samples, replace=False)
    return x[indices], y[indices]


def train_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> Tuple[Result, object, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    if config.classifier == "linear":
        clf = LinearSVC(
            C=config.c_value,
            dual=False,
            max_iter=6000,
            random_state=42,
            class_weight="balanced",
        )
    elif config.classifier == "rbf":
        clf = SVC(
            C=config.c_value,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported classifier type: {config.classifier}")

    clf.fit(x_train_scaled, y_train)

    val_pred = clf.predict(x_val_scaled)
    test_pred = clf.predict(x_test_scaled)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    cm = confusion_matrix(y_test, test_pred)

    return (
        Result(
            config=config,
            val_accuracy=float(val_acc),
            val_macro_f1=float(val_f1),
            test_accuracy=float(test_acc),
            test_macro_f1=float(test_f1),
            confusion_matrix=cm.tolist(),
        ),
        clf,
        scaler,
    )


def build_search_space() -> List[Config]:
    configs: List[Config] = []
    preprocess_modes = ["none", "equalize", "clahe"]
    orientations = [8, 9, 12]
    pixels = [(6, 6), (8, 8)]
    blocks = [(2, 2), (3, 3)]
    linear_c = [0.5, 1.0, 2.0]
    rbf_c = [1.0, 5.0]

    for mode in preprocess_modes:
        for ori in orientations:
            for ppc in pixels:
                for cpb in blocks:
                    for c in linear_c:
                        configs.append(
                            Config(
                                preprocess=mode,
                                orientations=ori,
                                pixels_per_cell=ppc,
                                cells_per_block=cpb,
                                classifier="linear",
                                c_value=c,
                            )
                        )
                    for c in rbf_c:
                        configs.append(
                            Config(
                                preprocess=mode,
                                orientations=ori,
                                pixels_per_cell=ppc,
                                cells_per_block=cpb,
                                classifier="rbf",
                                c_value=c,
                            )
                        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune HOG + SVM on FER2013 folder dataset.")
    parser.add_argument("--data-path", default="data/fer2013", help="Path to FER2013 folder dataset.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on train samples.")
    parser.add_argument("--max-val", type=int, default=None, help="Optional cap on validation samples.")
    parser.add_argument("--max-test", type=int, default=None, help="Optional cap on test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    train_dataset = FER2013Dataset(args.data_path, split="Training")
    val_dataset = FER2013Dataset(args.data_path, split="PublicTest")
    test_dataset = FER2013Dataset(args.data_path, split="PrivateTest")

    search_space = build_search_space()
    print(f"Total configs to evaluate: {len(search_space)}")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    best_result: Optional[Result] = None
    best_clf = None
    best_scaler = None

    feature_cache: Dict[Tuple[str, int, Tuple[int, int], Tuple[int, int], str], Tuple[np.ndarray, np.ndarray]] = {}

    for idx, cfg in enumerate(search_space, start=1):
        print(
            f"[{idx}/{len(search_space)}] mode={cfg.preprocess} ori={cfg.orientations} "
            f"ppc={cfg.pixels_per_cell} cpb={cfg.cells_per_block} clf={cfg.classifier} C={cfg.c_value}"
        )

        train_key = (cfg.preprocess, cfg.orientations, cfg.pixels_per_cell, cfg.cells_per_block, "train")
        val_key = (cfg.preprocess, cfg.orientations, cfg.pixels_per_cell, cfg.cells_per_block, "val")
        test_key = (cfg.preprocess, cfg.orientations, cfg.pixels_per_cell, cfg.cells_per_block, "test")

        if train_key not in feature_cache:
            feature_cache[train_key] = extract_hog_features(
                train_dataset,
                cfg.preprocess,
                cfg.orientations,
                cfg.pixels_per_cell,
                cfg.cells_per_block,
            )
        if val_key not in feature_cache:
            feature_cache[val_key] = extract_hog_features(
                val_dataset,
                cfg.preprocess,
                cfg.orientations,
                cfg.pixels_per_cell,
                cfg.cells_per_block,
            )
        if test_key not in feature_cache:
            feature_cache[test_key] = extract_hog_features(
                test_dataset,
                cfg.preprocess,
                cfg.orientations,
                cfg.pixels_per_cell,
                cfg.cells_per_block,
            )

        x_train, y_train = feature_cache[train_key]
        x_val, y_val = feature_cache[val_key]
        x_test, y_test = feature_cache[test_key]

        x_train, y_train = maybe_subsample(x_train, y_train, args.max_train, seed=args.seed)
        x_val, y_val = maybe_subsample(x_val, y_val, args.max_val, seed=args.seed + 1)
        x_test, y_test = maybe_subsample(x_test, y_test, args.max_test, seed=args.seed + 2)

        result, clf, scaler = train_and_eval(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            cfg,
        )

        all_rows.append(
            {
                "preprocess": cfg.preprocess,
                "orientations": cfg.orientations,
                "pixels_per_cell": str(cfg.pixels_per_cell),
                "cells_per_block": str(cfg.cells_per_block),
                "classifier": cfg.classifier,
                "c_value": cfg.c_value,
                "val_accuracy": result.val_accuracy,
                "val_macro_f1": result.val_macro_f1,
                "test_accuracy": result.test_accuracy,
                "test_macro_f1": result.test_macro_f1,
            }
        )

        if best_result is None or (result.val_macro_f1, result.val_accuracy) > (
            best_result.val_macro_f1,
            best_result.val_accuracy,
        ):
            best_result = result
            best_clf = clf
            best_scaler = scaler

    if best_result is None or best_clf is None or best_scaler is None:
        raise RuntimeError("No configuration evaluated successfully.")

    csv_path = reports_dir / "hog_svm_tuning_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "preprocess,orientations,pixels_per_cell,cells_per_block,classifier,c_value,val_accuracy,val_macro_f1,test_accuracy,test_macro_f1\n"
        )
        for row in all_rows:
            f.write(
                f"{row['preprocess']},{row['orientations']},{row['pixels_per_cell']},{row['cells_per_block']},"
                f"{row['classifier']},{row['c_value']},{row['val_accuracy']:.6f},{row['val_macro_f1']:.6f},"
                f"{row['test_accuracy']:.6f},{row['test_macro_f1']:.6f}\n"
            )

    best_json = {
        "best_config": {
            "preprocess": best_result.config.preprocess,
            "orientations": best_result.config.orientations,
            "pixels_per_cell": list(best_result.config.pixels_per_cell),
            "cells_per_block": list(best_result.config.cells_per_block),
            "classifier": best_result.config.classifier,
            "c_value": best_result.config.c_value,
        },
        "metrics": {
            "val_accuracy": best_result.val_accuracy,
            "val_macro_f1": best_result.val_macro_f1,
            "test_accuracy": best_result.test_accuracy,
            "test_macro_f1": best_result.test_macro_f1,
            "confusion_matrix": best_result.confusion_matrix,
        },
    }

    best_json_path = reports_dir / "hog_svm_best_config.json"
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(best_json, f, indent=2)

    artifact_payload = {
        "classifier": best_clf,
        "scaler": best_scaler,
        "hog_config": {
            "orientations": best_result.config.orientations,
            "pixels_per_cell": list(best_result.config.pixels_per_cell),
            "cells_per_block": list(best_result.config.cells_per_block),
        },
        "preprocess": best_result.config.preprocess,
        "class_order": EMOTIONS,
        "metrics": {
            "accuracy": best_result.test_accuracy,
            "macro_f1": best_result.test_macro_f1,
            "confusion_matrix": best_result.confusion_matrix,
            "val_accuracy": best_result.val_accuracy,
            "val_macro_f1": best_result.val_macro_f1,
        },
    }

    save_dir = Path("saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = save_dir / "hog_svm_artifact_tuned.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(artifact_payload, f)

    print("=" * 70)
    print("Best tuned HOG+SVM config")
    print(best_json)
    print("=" * 70)
    print(f"Saved tuned artifact: {artifact_path}")
    print(f"Saved tuning table:   {csv_path}")
    print(f"Saved best config:    {best_json_path}")


if __name__ == "__main__":
    main()
