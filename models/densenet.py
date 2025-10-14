import numpy as np
import torch
import torch.nn as nn
import mindspore
from mindspore import context
import troubleshooter as ts
import mindspore.nn as nn_ms
import mindspore.ops as ops

device=torch.device('cuda')

# 1. PyTorch DenseNet-121
class TorchBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inter = growth_rate * self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels).to(device)
        self.conv1 = nn.Conv2d(in_channels, inter, 1, bias=False).to(device)
        self.bn2 = nn.BatchNorm2d(inter).to(device)
        self.conv2 = nn.Conv2d(inter, growth_rate, 3, padding=1, bias=False).to(device)

    def forward(self, x):
        out = self.conv1(nn.functional.relu(self.bn1(x))).to(device)
        out = self.conv2(nn.functional.relu(self.bn2(out))).to(device)
        return torch.cat([x, out], 1).to(device)

class TorchTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels).to(device)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False).to(device)
        self.pool = nn.AvgPool2d(2, 2).to(device)

    def forward(self, x):
        x = self.conv(nn.functional.relu(self.bn(x))).to(device)
        return self.pool(x).to(device)

class TorchDenseNet(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16)):
        super().__init__()
        n_init = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, n_init, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(n_init),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        ).to(device)

        n_feat = n_init
        for i, n_layers in enumerate(block_config):
            dense_blk = nn.Sequential(
                *[TorchBottleneck(n_feat + j * growth_rate, growth_rate) for j in range(n_layers)]
            ).to(device)
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
        x=x.to(device)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 2. MindSpore DenseNet-121
class MSBottleneck(nn_ms.Cell):
    expansion = 4
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inter = growth_rate * self.expansion
        self.bn1 = nn_ms.BatchNorm2d(in_channels)
        self.conv1 = nn_ms.Conv2d(in_channels, inter, 1, has_bias=False)
        self.bn2 = nn_ms.BatchNorm2d(inter)
        self.conv2 = nn_ms.Conv2d(inter, growth_rate, 3, padding=1, pad_mode='pad', has_bias=False)

    def construct(self, x):
        out = self.conv1(ops.relu(self.bn1(x)))
        out = self.conv2(ops.relu(self.bn2(out)))
        return ops.concat((x, out), 1)

class MSTransition(nn_ms.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn_ms.BatchNorm2d(in_channels)
        self.conv = nn_ms.Conv2d(in_channels, out_channels, 1, has_bias=False)
        self.pool = nn_ms.AvgPool2d(2, 2)

    def construct(self, x):
        x = self.conv(ops.relu(self.bn(x)))
        return self.pool(x)

class MSDenseNet(nn_ms.Cell):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16)):
        super().__init__()
        n_init = 64
        self.features = nn_ms.SequentialCell([
            nn_ms.Conv2d(3, n_init, 7, stride=2, padding=3, pad_mode='pad', has_bias=False),
            nn_ms.BatchNorm2d(n_init),
            nn_ms.ReLU(),
            nn_ms.MaxPool2d(3, stride=2, padding=1, pad_mode='pad')
        ])

        n_feat = n_init
        for i, n_layers in enumerate(block_config):
            dense_blk = nn_ms.SequentialCell(
                *[MSBottleneck(n_feat + j * growth_rate, growth_rate) for j in range(n_layers)]
            )
            self.features.append(dense_blk)
            n_feat += n_layers * growth_rate
            if i < len(block_config) - 1:
                trans = MSTransition(n_feat, n_feat // 2)
                self.features.append(trans)
                n_feat //= 2

        self.features.append(nn_ms.BatchNorm2d(n_feat))
        self.features.append(nn_ms.ReLU())
        self.features.append(nn_ms.AdaptiveAvgPool2d((7, 7)))
        self.classifier = nn_ms.Dense(n_feat * 7 * 7, num_classes)

    def construct(self, x):
        x = self.features(x)
        x = ops.flatten(x)
        return self.classifier(x)
