import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

HOG_CONFIG: Dict[str, object] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
}


def extract_hog_features(dataset: FER2013Dataset) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = np.asarray(dataset.emotions, dtype=np.int64)

    for image in dataset.images:
        descriptor = hog(
            image,
            orientations=int(HOG_CONFIG["orientations"]),
            pixels_per_cell=HOG_CONFIG["pixels_per_cell"],
            cells_per_block=HOG_CONFIG["cells_per_block"],
            visualize=False,
            feature_vector=True,
        )
        features.append(descriptor)

    return np.asarray(features, dtype=np.float32), labels


def main() -> None:
    csv_path = "data/fer2013"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    train_dataset = FER2013Dataset(csv_path, split="Training")
    test_dataset = FER2013Dataset(csv_path, split="PrivateTest")

    print("Extracting HOG features from training set...")
    x_train, y_train = extract_hog_features(train_dataset)
    print("Extracting HOG features from test set...")
    x_test, y_test = extract_hog_features(test_dataset)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = LinearSVC(dual=False, max_iter=4000, random_state=42)
    clf.fit(x_train_scaled, y_train)

    y_pred = clf.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("HOG + SVM artifact training complete")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print("=" * 60)

    save_dir = Path("saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = save_dir / "hog_svm_artifact.pkl"

    payload = {
        "classifier": clf,
        "scaler": scaler,
        "hog_config": {
            "orientations": HOG_CONFIG["orientations"],
            "pixels_per_cell": list(HOG_CONFIG["pixels_per_cell"]),
            "cells_per_block": list(HOG_CONFIG["cells_per_block"]),
        },
        "class_order": EMOTIONS,
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm.tolist(),
        },
    }

    with artifact_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved artifact to: {artifact_path}")


if __name__ == "__main__":
    main()
