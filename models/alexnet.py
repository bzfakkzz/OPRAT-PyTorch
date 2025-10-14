import numpy as np
import torch
import torch.nn as nn
import mindspore
from mindspore import context
import troubleshooter as ts
import mindspore.nn as nn_ms
import mindspore as ms

device=torch.device('cuda')

# PyTorch AlexNet-CIFAR10模型
class TorchAlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(TorchAlexNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.bn1   = nn.BatchNorm2d(96).to(device)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.bn2   = nn.BatchNorm2d(256).to(device)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1).to(device)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1).to(device)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.bn3   = nn.BatchNorm2d(256).to(device)

        self.flatten = nn.Flatten().to(device)
        self.fc1     = nn.Linear(1024, 4096).to(device)
        self.dropout1 = nn.Dropout(0.5).to(device)
        self.fc2     = nn.Linear(4096, 4096).to(device)
        self.dropout2 = nn.Dropout(0.5).to(device)
        self.fc3     = nn.Linear(4096, num_classes).to(device)

    def forward(self, x):
        x=x.to(device)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# MindSpore AlexNet-CIFAR10模型
class MindSporeAlexNetCIFAR10(nn_ms.Cell):
    def __init__(self, num_classes=10):
        super(MindSporeAlexNetCIFAR10, self).__init__()
        self.conv1 = nn_ms.Conv2d(3, 96, kernel_size=3, stride=2, pad_mode='same')
        self.pool1 = nn_ms.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.bn1   = nn_ms.BatchNorm2d(96)

        self.conv2 = nn_ms.Conv2d(96, 256, kernel_size=5, pad_mode='same')
        self.pool2 = nn_ms.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.bn2   = nn_ms.BatchNorm2d(256)

        self.conv3 = nn_ms.Conv2d(256, 384, kernel_size=3, pad_mode='same')
        self.conv4 = nn_ms.Conv2d(384, 384, kernel_size=3, pad_mode='same')
        self.conv5 = nn_ms.Conv2d(384, 256, kernel_size=3, pad_mode='same')
        self.pool3 = nn_ms.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.bn3   = nn_ms.BatchNorm2d(256)

        self.flatten = nn_ms.Flatten()
        self.fc1     = nn_ms.Dense(1024, 4096)
        self.dropout1 = nn_ms.Dropout(0.5)
        self.fc2     = nn_ms.Dense(4096, 4096)
        self.dropout2 = nn_ms.Dropout(0.5)
        self.fc3     = nn_ms.Dense(4096, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
