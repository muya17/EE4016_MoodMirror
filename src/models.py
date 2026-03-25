import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    The core building block for lightweight CNNs.
    Splits standard convolution into a Depthwise spatial convolution 
    and a Pointwise (1x1) channel-mixing convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 1. Depthwise Convolution (applies a single filter per input channel)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            padding=1, stride=stride, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        
        # 2. Pointwise Convolution (1x1 convolution to combine channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class CLCM(nn.Module):
    """
    Lightweight CNN Architecture based on the CLCM (2024) paper.
    Target: < 1.5M parameters, high efficiency for 48x48 grayscale images.
    """
    def __init__(self, num_classes=7):
        super(CLCM, self).__init__()
        
        # Initial Convolution: Note that in_channels=1 because FER2013 is grayscale
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Feature Extractor: A stack of Depthwise Separable Convolutions
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),   # Image size: 48x48 -> 24x24
            DepthwiseSeparableConv(64, 128, stride=2),  # Image size: 24x24 -> 12x12
            DepthwiseSeparableConv(128, 256, stride=2), # Image size: 12x12 -> 6x6
            DepthwiseSeparableConv(256, 256, stride=1), # Image size: 6x6 -> 6x6
        )

        # Global Average Pooling (Replaces parameter-heavy dense layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Classifier with Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # 50% probability to drop units (prevents overfitting)
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass data flow.
        """
        x = self.init_conv(x)
        x = self.features(x)
        x = self.pool(x)
        
        # Flatten the feature map from (Batch, Channels, 1, 1) to (Batch, Channels)
        x = torch.flatten(x, 1) 
        
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    """
    Standard CNN baseline. 
    (M3 will implement this later).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        pass

    def forward(self, x):
        return x