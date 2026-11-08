# ===================================================================================
#  YOLO11m Model Architecture in PyTorch
#  PyTorch Architecture Implementation (Deep Convolutional Object Detection Network)
# ===================================================================================

"""
Model Architecture Description:
YOLO11m model specialized for Object Detection tasks.
The model is trained to recognize 15 geometric color classes (circles, squares, and triangles in various colors).

Key architectural components:
1. Conv Module: Basic convolution block with Batch Normalization and Activation.
2. Bottleneck Module: Deep feature extraction block with optional residual connection.
3. C3k2 Module: High-efficiency feature fusion and distribution unit in YOLO11.
4. SPPF Module: Spatial Pyramid Pooling - Fast for multi-scale context aggregation.
5. C2PSA Module: Polarized Self-Attention mechanism for enhanced spatial focus.
6. Detect Head: Final prediction head for bounding box regression and classification.
"""

import torch                  # Core library for tensor computations and neural networks
import torch.nn as nn         # Neural network modules and layers in PyTorch

# ---------------------------------------------------------------------
# 1. Conv Module: 2D Convolution with Batch Normalization & Activation
# ---------------------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # c1: in_channels, c2: out_channels
        super().__init__()
        if p is None:
            p = k // 2  # Auto-compute padding to maintain spatial dimensions
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)  # 2D Convolution
        self.bn = nn.BatchNorm2d(c2)                                  # Batch Normalization
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity()) # SiLU activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # Forward pass: Conv -> BN -> Act


# ---------------------------------------------------------------------
# 2. Bottleneck: Deep Feature Processing Block with Residual Connection
# ---------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)                                        # Reduced internal channel dimension
        self.cv1 = Conv(c1, c_, k[0], 1)                        # 1st conv to compress channels
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)                   # 2nd conv for feature extraction
        self.add = shortcut and c1 == c2                        # Shortcut flag when channels match

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) # Residual addition if enabled


# ---------------------------------------------------------------------
# 3. C3k2 Module: Feature Fusion & Splitting Unit in YOLO11 Architecture
# ---------------------------------------------------------------------
class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)                                   # Reduced channels for compute efficiency
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)                  # Split input into two pathways
        self.cv2 = Conv((2 + n) * self.c, c2, 1)               # Combine concatenated pathways
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)) # Stacked Bottlenecks

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))                      # Split tensor into two feature branches
        y.extend(m(y[-1]) for m in self.m)                     # Pass second branch through Bottlenecks
        return self.cv2(torch.cat(y, 1))                       # Concatenate branches and apply final Conv


# ---------------------------------------------------------------------
# 4. SPPF Module: Spatial Pyramid Pooling - Fast
# ---------------------------------------------------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2                                           # Half input channels
        self.cv1 = Conv(c1, c_, 1, 1)                          # Initial feature compression
        self.cv2 = Conv(c_ * 4, c2, 1, 1)                      # Output fusion convolution
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) # Max pooling layer

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))              # Apply sequential pooling 3 times
        return self.cv2(torch.cat(y, 1))                       # Concatenate multi-scale context


# ---------------------------------------------------------------------
# 5. C2PSA Module: Polarized Self-Attention Mechanism
# ---------------------------------------------------------------------
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)                  # Split input channels
        self.cv2 = Conv(2 * self.c, c2, 1, 1)                  # Output fusion
        self.attn = nn.Sequential(                             # Lightweight self-attention network
            Conv(self.c, self.c, 1, 1),
            nn.Sigmoid()                                       # Attention weights in range [0, 1]
        )

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)                         # Split into direct & attended branches
        b = b * self.attn(b)                                   # Scale branch with attention weights
        return self.cv2(torch.cat((a, b), 1))                  # Re-merge and convolve


# ---------------------------------------------------------------------
# 6. Detect Head: Bounding Box Regression & Classification Head
# ---------------------------------------------------------------------
class Detect(nn.Module):
    def __init__(self, nc=15, ch=(256, 512, 1024)):           # nc: number of classes (15), ch: channels for P3, P4, P5
        super().__init__()
        self.nc = nc                                           # Number of target classes
        self.nl = len(ch)                                      # Number of detection scales (3 levels)
        self.reg_max = 16                                      # Coordinate distribution max (DFL)
        
        # Localization (cv2) and Classification (cv3) branches for each scale
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, x, 3), Conv(x, self.nc, 3), nn.Conv2d(self.nc, self.nc, 1)) for x in ch)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) # Concatenate box coords & class logits
        return x


# ---------------------------------------------------------------------
# 7. YOLO11mModel Class: Full Architecture Connecting Backbone to Head
# ---------------------------------------------------------------------
class YOLO11mModel(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        print(f"Constructing YOLO11m architecture for {num_classes} classes...")

        # --- Backbone ---
        self.layer0 = Conv(3, 64, 3, 2)                         # Initial downsampling P1/2
        self.layer1 = Conv(64, 128, 3, 2)                       # Downsampling P2/4
        self.layer2 = C3k2(128, 256, n=2, c3k=False)
        self.layer3 = Conv(256, 256, 3, 2)                      # Small object scale P3/8
        self.layer4 = C3k2(256, 512, n=2, c3k=False)
        self.layer5 = Conv(512, 512, 3, 2)                      # Medium object scale P4/16
        self.layer6 = C3k2(512, 512, n=2, c3k=True)
        self.layer7 = Conv(512, 1024, 3, 2)                     # Large object scale P5/32
        self.layer8 = C3k2(1024, 1024, n=2, c3k=True)
        self.layer9 = SPPF(1024, 1024, 5)                       # Spatial pyramid pooling
        self.layer10 = C2PSA(1024, 1024, n=2)                   # Polarized self-attention

        # --- Detection Head ---
        self.head = Detect(nc=num_classes, ch=(256, 512, 1024)) # Final detection head for 15 classes

    def forward(self, x):
        # 1. Forward through Backbone
        x = self.layer0(x)
        x = self.layer1(x)
        p2 = self.layer2(x)
        x = self.layer3(p2)
        p3 = self.layer4(x)                                    # Small object features
        x = self.layer5(p3)
        p4 = self.layer6(x)                                    # Medium object features
        x = self.layer7(p4)
        x = self.layer8(x)
        x = self.layer9(x)
        p5 = self.layer10(x)                                   # Large object features

        # 2. Forward through Detection Head
        out = self.head([p3, p4, p5])
        return out


if __name__ == "__main__":
    model = YOLO11mModel(num_classes=15)                        # Instantiate model test for 15 classes
    print("YOLO11m architecture successfully instantiated.")
