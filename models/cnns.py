import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock1x1(nn.Module):
    """Basic Resnet block but instead of 3x3 convs we use 1x1 convs"""
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride = 1,
                 downsample = None,
                 groups = 1,
                 base_width = 64,
                 dilation = 1,
                 norm_layer = None):
        
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    


class Resnet1x1(ResNet):

    def __init__(self, 
                 n_classes, 
                 n_bands,
                 *args,
                 **kwargs
                ):
        
        super().__init__(BasicBlock1x1, [2, 2, 2, 2], num_classes=n_classes)

        self.n_bands = n_bands
    
        self.conv1 = nn.Conv2d(n_bands, 
                                      64, 
                                      kernel_size=3, 
                                      stride=1, 
                                      padding=1, 
                                      bias=False)
        
        total_feats = 512
        
        self.fc = nn.Linear(total_feats, n_classes)


class SmallSequentualCNN(nn.Module):
    def __init__(self, 
                 n_classes, 
                 n_bands,
                 *args,
                 **kwargs
                ):
        super(SmallSequentualCNN, self).__init__()

        # Input shape: (batch_size, 12, 6, 6)
        self.conv1 = nn.Conv2d(n_bands, 16, kernel_size=1, padding=1)  # (batch_size, 32, 6, 6)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, padding=1)  # (batch_size, 64, 6, 6)

        self.fc1 = nn.Linear(3200, 256) 
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
