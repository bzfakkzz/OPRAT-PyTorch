import numpy as np
import torch
import torch.nn as nn
import troubleshooter as ts


# del. PyTorch DenseNet-121
class TorchBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inter = growth_rate * self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(nn.functional.relu(self.bn1(x)))
        out = self.conv2(nn.functional.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class TorchTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv(nn.functional.relu(self.bn(x)))
        return self.pool(x)

class TorchDenseNet(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16)):
        super().__init__()
        n_init = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, n_init, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(n_init),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        n_feat = n_init
        for i, n_layers in enumerate(block_config):
            dense_blk = nn.Sequential(
                *[TorchBottleneck(n_feat + j * growth_rate, growth_rate) for j in range(n_layers)]
            )
            self.features.add_module(f"dense_{i}", dense_blk)
            n_feat += n_layers * growth_rate
            if i < len(block_config) - 1:
                trans = TorchTransition(n_feat, n_feat // 2)
                self.features.add_module(f"trans_{i}", trans)
                n_feat //= 2

        self.features.add_module("final_bn", nn.BatchNorm2d(n_feat))
        self.features.add_module("relu_end", nn.ReLU(inplace=True))
        self.features.add_module("avgpool", nn.AdaptiveAvgPool2d((7, 7)))

        self.classifier = nn.Linear(n_feat * 7 * 7, num_classes)

    def forward(self, x):
        x=x
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
