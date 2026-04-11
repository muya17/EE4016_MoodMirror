import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split='Training', transform=None):
        """
        Custom Dataset for FER2013.
        :param csv_file: Path to the fer2013.csv file.
        :param split: 'Training', 'PublicTest' (Validation), or 'PrivateTest' (Test).
        :param transform: PyTorch transforms for data augmentation and preprocessing.
        """
        print(f"Loading {split} data...")
        
        # 1. Read the entire CSV file
        df = pd.read_csv(csv_file)
        
        # 2. Filter data based on the 'Usage' column
        df = df[df['Usage'] == split]
        
        self.transform = transform
        self.emotions = df['emotion'].values
        
        # 3. Convert space-separated pixel strings into 48x48 matrices
        self.images = []
        for pixel_str in df['pixels'].values:
            # Split string by space and convert to 8-bit unsigned integer (0-255)
            pixels = np.fromstring(pixel_str, sep=' ', dtype=np.uint8)
            # Reshape into a 48x48 2D matrix
            image = pixels.reshape(48, 48)
            self.images.append(image)
            
        print(f"{split} data loaded successfully! Total images: {len(self.images)}\n")

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.emotions[idx]
        
        # Convert numpy array to PIL Image (required by most torchvision transforms)
        image = Image.fromarray(image)
        
        # Apply data augmentation and preprocessing
        if self.transform:
            image = self.transform(image)
            
        # Ensure label is a long tensor (required by PyTorch CrossEntropyLoss)
        label = torch.tensor(label, dtype=torch.long)
            
        return image, label


def get_dataloaders(csv_path, batch_size=64):
    """
    Main interface to get DataLoaders for Training, Validation, and Testing.
    """
    # Data augmentation logic (Only applied to the training set)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),                       # Converts to Tensor and scales pixels to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Standardization for 1-channel grayscale image
    ])

    # Validation and Test sets MUST NOT have data augmentation!
    # Only tensor conversion and standardization.
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Instantiate datasets
    train_dataset = FER2013Dataset(csv_path, split='Training', transform=train_transforms)
    val_dataset = FER2013Dataset(csv_path, split='PublicTest', transform=test_transforms)
    test_dataset = FER2013Dataset(csv_path, split='PrivateTest', transform=test_transforms)

    # Wrap with DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Local testing block: executed only when running this script directly
    test_csv_path = "data/fer2013.csv" 
    
    if os.path.exists(test_csv_path):
        print("Dataset found. Testing DataLoaders...")
        train_loader, val_loader, test_loader = get_dataloaders(test_csv_path, batch_size=16)
        
        # Try fetching one batch to verify tensor shapes
        images, labels = next(iter(train_loader))
        print("\nSuccessfully fetched one batch!")
        print(f"Images shape: {images.shape}")  # Expected: [16, 1, 48, 48]
        print(f"Labels shape: {labels.shape}")  # Expected: [16]
    else:
        print(f"Warning: Dataset not found at {test_csv_path}. Please ensure it is downloaded and placed correctly.")