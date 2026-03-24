import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split='Training', transform=None):
        """
        解析 FER2013 数据集
        split 选项: 'Training' (训练集), 'PublicTest' (验证集), 'PrivateTest' (测试集)
        """
        print(f"正在加载 {split} 数据...")
        
        # 1. 读取整个 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 2. 根据 Usage 列筛选数据
        df = df[df['Usage'] == split]
        
        self.transform = transform
        self.emotions = df['emotion'].values
        
        # 3. 将长长的字符串像素转换为 48x48 的矩阵
        self.images = []
        for pixel_str in df['pixels'].values:
            # 用空格分割字符串，转为 8位无符号整数 (0-255)
            pixels = np.fromstring(pixel_str, sep=' ', dtype=np.uint8)
            # 重塑为 48x48 的二维矩阵
            image = pixels.reshape(48, 48)
            self.images.append(image)
            
        print(f"{split} 数据加载完成！共 {len(self.images)} 张图片。\n")

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.emotions[idx]
        
        # 将 numpy 矩阵转换为 PIL Image，这是 torchvision 数据增强的基础
        image = Image.fromarray(image)
        
        # 应用数据增强和预处理
        if self.transform:
            image = self.transform(image)
            
        # 确保标签是长整型（PyTorch 计算交叉熵损失的要求）
        label = torch.tensor(label, dtype=torch.long)
            
        return image, label


def get_dataloaders(csv_path, batch_size=64):
    """
    暴露给外部调用的主接口，返回 Train, Val, Test 的 DataLoader
    """
    # 按照 Proposal 4.3 设计的数据增强逻辑 (仅用于训练集)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),           # 随机水平翻转
        transforms.RandomRotation(10),                    # 小角度旋转 (±10°)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 随机平移 (对应 crop/shift)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),     # 轻微亮度/对比度调整
        transforms.ToTensor(),                            # 转为 Tensor 并归一化到 [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])       # 标准化 (灰度图单通道)
    ])

    # 验证集和测试集绝对不能做数据增强！只能做张量转换和标准化
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 实例化数据集
    train_dataset = FER2013Dataset(csv_path, split='Training', transform=train_transforms)
    val_dataset = FER2013Dataset(csv_path, split='PublicTest', transform=test_transforms)
    test_dataset = FER2013Dataset(csv_path, split='PrivateTest', transform=test_transforms)

    # 包装成 DataLoader，负责批量打包和打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 本地测试代码：直接运行这个脚本时会执行这里
    test_csv_path = "../data/fer2013.csv" 
    
    import os
    if os.path.exists(test_csv_path):
        print("找到数据集，开始测试 DataLoader...")
        train_loader, val_loader, test_loader = get_dataloaders(test_csv_path, batch_size=16)
        
        # 尝试抽取一个 Batch 看一下形状
        images, labels = next(iter(train_loader))
        print(f"\n成功抽取一个 Batch!")
        print(f"Images shape: {images.shape}")  # 期望输出: [16, 1, 48, 48]
        print(f"Labels shape: {labels.shape}")  # 期望输出: [16]
    else:
        print(f"警告：未在 {test_csv_path} 找到数据集。请确保数据集已下载并放对位置。")