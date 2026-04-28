import os
from pathlib import Path
from typing import List, Tuple

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
    """Split class items into train/val while preserving class distribution."""
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
    train_items = [item for idx, item in enumerate(items) if idx not in val_indices]
    val_items = [item for idx, item in enumerate(items) if idx in val_indices]
    return train_items, val_items


class FER2013Dataset(Dataset):
    """
    Flexible FER2013 dataset that works with:
    - The original fer2013.csv file
    - A folder structure with train/ and test/ subfolders (e.g., Kaggle dataset)
    """
    def __init__(self, source_path, split='Training', transform=None, val_fraction=0.1, seed=42):
        """
        Args:
            source_path: Path to CSV file OR root folder containing train/ and test/
            split: For CSV: 'Training', 'PublicTest', 'PrivateTest'
                   For folder: 'train', 'val' (from train split), 'test'
            transform: torchvision transforms
            val_fraction: Fraction of training data to use as validation (only when split='val')
            seed: Random seed for reproducibility
        """
        print(f"Loading {split} data...")
        self.transform = transform
        self.images: List[np.ndarray] = []
        self.emotions: List[int] = []

        source = Path(source_path)
        normalized_split = split.strip().lower()

        # Detect if source is a CSV file or a folder
        if source.is_file() and source.suffix.lower() == ".csv":
            self._load_from_csv(source, split)
        else:
            # Assume folder structure (Kaggle style)
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

    def _load_from_folder(self, root: Path, split: str, val_fraction: float, seed: int) -> None:
        # Locate train and test directories
        train_dir = root / "train" if (root / "train").is_dir() else root
        test_dir = root / "test" if (root / "test").is_dir() else root

        if split in {"training", "train"}:
            # Load training subset and keep validation disjoint
            self._load_folder_subset(train_dir, use_validation=False,
                                     val_fraction=val_fraction, seed=seed)
        elif split in {"validation", "val", "valid"}:
            # Load validation set (split from training)
            self._load_folder_subset(train_dir, use_validation=True,
                                     val_fraction=val_fraction, seed=seed)
        elif split in {"test", "testing", "privatetest"}:
            self._load_entire_folder(test_dir)
        else:
            raise ValueError(f"Unsupported split for folder dataset: {split}")

    def _load_folder_subset(self, folder_root: Path, use_validation: bool,
                            val_fraction: float, seed: int) -> None:
        """Load either train or validation subset from a folder of class subfolders."""
        for class_index, emotion in enumerate(EMOTIONS):
            class_dir = folder_root / emotion
            if not class_dir.is_dir():
                continue

            class_items: List[np.ndarray] = []
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and _is_image_file(image_path):
                    class_items.append(_load_grayscale_array(image_path))

            train_items, val_items = _split_classwise(class_items, val_fraction=val_fraction,
                                                      seed=seed + class_index)
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


def get_dataloaders(data_path, batch_size=256, augment=True, num_workers=4, pin_memory=True, val_fraction=0.1):
    """
    Create train, validation, and test DataLoaders for FER2013.

    Args:
        data_path (str): Path to either:
            - the fer2013.csv file, OR
            - the root folder containing 'train' and 'test' subfolders (Kaggle style)
        batch_size (int): Batch size
        augment (bool): Apply data augmentation to training set
        num_workers (int): Number of subprocesses for data loading
        pin_memory (bool): Pin memory for faster GPU transfer
        val_fraction (float): Fraction of training data to use as validation (folder mode only)

    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_transform = aug_transform if augment else base_transform
    val_transform = base_transform
    test_transform = base_transform

    path_obj = Path(data_path)

    # Detect whether we are using CSV or folder
    if path_obj.is_file() and path_obj.suffix.lower() == ".csv":
        # CSV mode
        train_dataset = FER2013Dataset(data_path, split='Training', transform=train_transform)
        val_dataset = FER2013Dataset(data_path, split='PublicTest', transform=val_transform)
        test_dataset = FER2013Dataset(data_path, split='PrivateTest', transform=test_transform)
    else:
        # Folder mode (Kaggle)
        # Training set with augmentation (or without)
        train_dataset = FER2013Dataset(data_path, split='train', transform=train_transform,
                                       val_fraction=val_fraction, seed=42)
        # Validation set is created from the same training folder, no augmentation
        val_dataset = FER2013Dataset(data_path, split='val', transform=val_transform,
                                     val_fraction=val_fraction, seed=42)
        test_dataset = FER2013Dataset(data_path, split='test', transform=test_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# Example usage with kagglehub
if __name__ == "__main__":
    import kagglehub

    # Download dataset using kagglehub
    print("Downloading FER2013 dataset...")
    dataset_path = kagglehub.dataset_download("msambare/fer2013")
    print(f"Dataset downloaded to: {dataset_path}")

    # Create dataloaders (with augmentation)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=dataset_path,
        batch_size=64,
        augment=True
    )

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nTrain batch shape: {images.shape}, labels: {labels.shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")