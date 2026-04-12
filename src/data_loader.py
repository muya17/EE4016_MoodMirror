import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _load_grayscale_array(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)


def _split_classwise(items: List[np.ndarray], val_fraction: float, seed: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not items:
        return [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)

    val_size = int(round(len(items) * val_fraction))
    if val_size == 0 and len(items) > 1:
        val_size = 1
    if val_size >= len(items):
        val_size = len(items) - 1

    val_indices = set(indices[:val_size])
    train_items = [item for index, item in enumerate(items) if index not in val_indices]
    val_items = [item for index, item in enumerate(items) if index in val_indices]
    return train_items, val_items


class FER2013Dataset(Dataset):
    def __init__(self, source_path, split='Training', transform=None, val_fraction=0.1, seed=42):
        """
        Custom Dataset for FER2013.
        Supports either the classic fer2013.csv file or the folder dataset layout.
        :param source_path: Path to the fer2013.csv file or the dataset folder.
        :param split: 'Training', 'PublicTest' (Validation), or 'PrivateTest' (Test).
        :param transform: PyTorch transforms for data augmentation and preprocessing.
        """
        print(f"Loading {split} data...")

        self.transform = transform
        self.images: List[np.ndarray] = []
        self.emotions: List[int] = []

        source = Path(source_path)
        normalized_split = split.strip().lower()

        if source.is_file() and source.suffix.lower() == ".csv":
            self._load_from_csv(source, split)
        else:
            self._load_from_folder(source, normalized_split, val_fraction=val_fraction, seed=seed)

        print(f"{split} data loaded successfully! Total images: {len(self.images)}\n")

    def _load_from_csv(self, csv_file: Path, split: str) -> None:
        df = pd.read_csv(csv_file)
        df = df[df['Usage'] == split]

        self.emotions = df['emotion'].values.tolist()

        for pixel_str in df['pixels'].values:
            pixels = np.fromstring(pixel_str, sep=' ', dtype=np.uint8)
            image = pixels.reshape(48, 48)
            self.images.append(image)

    def _load_from_folder(self, source: Path, split: str, val_fraction: float, seed: int) -> None:
        if source.name.lower() in {"train", "test", "validation", "val"}:
            root = source.parent
        else:
            root = source

        train_root = root / "train" if (root / "train").is_dir() else root
        test_root = root / "test" if (root / "test").is_dir() else root

        if split in {"training", "train"}:
            self._load_folder_subset(train_root, use_validation=False, val_fraction=val_fraction, seed=seed)
            return

        if split in {"publictest", "validation", "val", "valid"}:
            self._load_folder_subset(train_root, use_validation=True, val_fraction=val_fraction, seed=seed)
            return

        if split in {"privatetest", "test", "testing"}:
            self._load_entire_folder(test_root)
            return

        raise ValueError(f"Unsupported split: {split}")

    def _load_folder_subset(self, folder_root: Path, use_validation: bool, val_fraction: float, seed: int) -> None:
        for class_index, emotion in enumerate(EMOTIONS):
            class_dir = folder_root / emotion
            if not class_dir.is_dir():
                continue

            class_items: List[np.ndarray] = []
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and _is_image_file(image_path):
                    class_items.append(_load_grayscale_array(image_path))

            train_items, val_items = _split_classwise(class_items, val_fraction=val_fraction, seed=seed + class_index)
            selected_items = val_items if use_validation else train_items

            self.images.extend(selected_items)
            self.emotions.extend([class_index] * len(selected_items))

    def _load_entire_folder(self, folder_root: Path) -> None:
        for class_index, emotion in enumerate(EMOTIONS):
            class_dir = folder_root / emotion
            if not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and _is_image_file(image_path):
                    self.images.append(_load_grayscale_array(image_path))
                    self.emotions.append(class_index)

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.emotions[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_dataloaders(csv_path, batch_size=64, num_workers=2, pin_memory=False):
    """
    Main interface to get DataLoaders for Training, Validation, and Testing.
    """
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = FER2013Dataset(csv_path, split='Training', transform=train_transforms)
    val_dataset = FER2013Dataset(csv_path, split='PublicTest', transform=test_transforms)
    test_dataset = FER2013Dataset(csv_path, split='PrivateTest', transform=test_transforms)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    test_data_path = "data/fer2013"

    if os.path.exists(test_data_path):
        print("Dataset found. Testing DataLoaders...")
        train_loader, val_loader, test_loader = get_dataloaders(test_data_path, batch_size=16)

        images, labels = next(iter(train_loader))
        print("\nSuccessfully fetched one batch!")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    else:
        print(f"Warning: Dataset not found at {test_data_path}. Please ensure it is downloaded and placed correctly.")