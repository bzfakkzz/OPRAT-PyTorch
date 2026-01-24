import numpy as np
import torch
import torch.nn as nn、
import troubleshooter as ts、


# PyTorch AlexNet-CIFAR10模型
class TorchAlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(TorchAlexNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(1024, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2     = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3     = nn.Linear(4096, num_classes)

    def forward(self, x):
        x=x
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
