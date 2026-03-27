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
    

# ==============================================================================
# TESTING CODE (for M2 verification)
# ==============================================================================
if __name__ == "__main__":
    import time
    print("\n" + "="*50)
    print("STARTING CLCM (2024) MODEL VERIFICATION")
    print("="*50)

    # 1. Instantiate the model
    try:
        model = CLCM(num_classes=7)
        print("\n[PASS 1/3] Successfully instantiated CLCM model class.")
    except Exception as e:
        print(f"\n[FAIL 1/3] Failed to instantiate CLCM model. Error: {e}")
        exit()

    # 2. Verify Parameter Count (Crucial for M2 goal < 1.5M)
    # PyTorch idiom to count only trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[PASS 2/3] Verification of Model Complexity:")
    print(f"Total trainable parameters: {total_params:,}")

    LIMIT = 1_500_000
    if total_params < LIMIT:
        print(f"SUCCESS: Parameter count is well under the 1.5M budget")
    else:
        print(f"WARNING: Parameter count exceeds 1.5M! Needs optimization.")

    # 3. Test Forward Pass with Dummy Data (Verifies input shapes & tensor flow)
    print("\n[PASS 3/3] Testing forward pass with dummy input...")
    
    # Simulate FER2013 input: 1 Batch, 1 Channel (grayscale), 48x48 resolution
    dummy_input = torch.randn(1, 1, 48, 48)
    
    start_time = time.time()
    try:
        output = model(dummy_input)
        end_time = time.time()
        
        print(f"Successfully completed forward pass in {(end_time - start_time)*1000:.2f}ms.")
        
        # Output shape should be (BatchSize, NumClasses) -> (1, 7)
        print(f"Simulated Input shape : {dummy_input.shape}")
        print(f"Model Output shape   : {output.shape}")
        
        if output.shape == torch.Size([1, 7]):
            print(f"SUCCESS: Tensor shapes flowing correctly")
        else:
            print(f"WARNING: Output shape mismatch. Expected [1, 7].")

    except Exception as e:
        print(f"Failed tensor flow during forward pass. Error: {e}")

    print("\n" + "="*50)
    print("MODEL VERIFICATION COMPLETE")
    print("="*50 + "\n")