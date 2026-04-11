from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    A simple CNN architecture for the M3 deliverable.
    Designed to achieve ≥68% validation accuracy on FER2013.
    """
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        
        # Conv Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # 48x48 -> 24x24
        self.dropout1 = nn.Dropout(0.25)
        
        # Conv Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 24x24 -> 12x12
        self.dropout2 = nn.Dropout(0.25)
        
        # Conv Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 12x12 -> 6x6
        self.dropout3 = nn.Dropout(0.25)
        
        # Conv Block 4: 128 -> 256 channels (optional for better accuracy)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # 6x6 -> 3x3
        self.dropout4 = nn.Dropout(0.25)
        
        # Global Average Pooling (replaces large FC layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Head
        self.fc = nn.Linear(256, num_classes)
        self.dropout_fc = nn.Dropout(0.5)


    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # FC Head
        x = self.dropout_fc(F.relu(x))
        x = self.fc(x)
        
        return x
    
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution for LiteCNN.
    Reduces parameters while maintaining accuracy.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection.
    Helps gradient flow and improves accuracy.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        
        self.relu = nn.ReLU6(inplace=True)
        
        # Skip connection if channels or size change
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.skip is not None:
            residual = self.skip(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class LiteCNN(nn.Module):
    """
    An even more lightweight CNN architecture for experimentation.
    Target: < 1.5M parameters, achieve ≥68% validation accuracy.
    """
    def __init__(self, num_classes=7):
        super(LiteCNN, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Feature extraction with residual blocks
        # Block 1: 48x48 -> 24x24
        self.block1 = self._make_layer(32, 48, 2, stride=2)
        
        # Block 2: 24x24 -> 12x12
        self.block2 = self._make_layer(48, 96, 2, stride=2)
        
        # Block 3: 12x12 -> 6x6
        self.block3 = self._make_layer(96, 192, 3, stride=2)
        
        # Block 4: 6x6 -> 3x3
        self.block4 = self._make_layer(192, 384, 2, stride=2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==============================================================================
# TESTING CODE (for M3 verification)
# ==============================================================================
if __name__ == "__main__":
    import time
    import torch.nn.functional as F
    
    print("\n" + "="*50)
    print("TESTING M3 MODELS (SimpleCNN & CLCM)")
    print("="*50)
    
    # Test SimpleCNN
    print("\n--- SimpleCNN ---")
    simple_model = SimpleCNN(num_classes=7)
    simple_params = count_parameters(simple_model)
    print(f"Parameters: {simple_params:,}")

    # Test LiteCNN ← NEW
    print("\n--- LiteCNN ---")
    lite_model = LiteCNN(num_classes=7)
    lite_params = count_parameters(lite_model)
    print(f"Parameters: {lite_params:,}")
    
    # Test CLCM
    print("\n--- CLCM (M2) ---")
    clcm_model = CLCM(num_classes=7)
    clcm_params = count_parameters(clcm_model)
    print(f"Parameters: {clcm_params:,}")
    
    # Verify parameter counts
    print("\n--- Parameter Budget Check ---")
    if lite_params < 1_500_000:
        print(f"✅ LiteCNN meets budget (<1.5M): {lite_params:,}")
    else:
        print(f"❌ LiteCNN exceeds budget: {lite_params:,}")

    if clcm_params < 1_500_000:
        print(f"✅ CLCM meets budget (<1.5M): {clcm_params:,}")
    else:
        print(f"❌ CLCM exceeds budget: {clcm_params:,}")
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    dummy_input = torch.randn(1, 1, 48, 48)
    
    simple_output = simple_model(dummy_input)
    print(f"SimpleCNN output shape: {simple_output.shape}")
    
    lite_output = lite_model(dummy_input)
    print(f"LiteCNN output shape: {lite_output.shape}")
    
    clcm_output = clcm_model(dummy_input)
    print(f"CLCM output shape: {clcm_output.shape}")
    
    # Test inference latency (CPU)
    print("\n--- CPU Latency Test ---")
    simple_model.eval()
    lite_model.eval()
    clcm_model.eval()
    
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = simple_model(dummy_input)
        simple_latency = (time.time() - start) / 100 * 1000
        print(f"SimpleCNN avg latency: {simple_latency:.2f}ms")
        
        start = time.time()
        for _ in range(100):
            _ = lite_model(dummy_input)
        lite_latency = (time.time() - start) / 100 * 1000
        print(f"LiteCNN avg latency: {lite_latency:.2f}ms")
        
        start = time.time()
        for _ in range(100):
            _ = clcm_model(dummy_input)
        clcm_latency = (time.time() - start) / 100 * 1000
        print(f"CLCM avg latency: {clcm_latency:.2f}ms")
    
    print("\n" + "="*50)
    print("MODEL TEST COMPLETE")
    print("="*50 + "\n")