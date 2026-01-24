import numpy as np
import torch
import torch.nn as nn
import mindspore as ms
from mindspore import context
import troubleshooter as ts
import mindspore.nn as nn_ms

# ---------------------------- del. PyTorch LeNet5 ----------------------------
class TorchLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(TorchLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 120)  # 16*7*7=784
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x=x
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ---------------------------- 2. MindSpore LeNet5 ----------------------------
class MindSporeLeNet5(nn_ms.Cell):
    def __init__(self, num_classes=10):
        super(MindSporeLeNet5, self).__init__()
        self.conv1 = nn_ms.Conv2d(1, 6, kernel_size=5, padding=2, pad_mode='pad')
        self.pool1 = nn_ms.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn_ms.Conv2d(6, 16, kernel_size=5, padding=2, pad_mode='pad')
        self.pool2 = nn_ms.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn_ms.Flatten()
        self.fc1 = nn_ms.Dense(784, 120)  # 16*7*7=784
        self.dropout = nn_ms.Dropout(keep_prob=0.5)
        self.fc2 = nn_ms.Dense(120, 84)
        self.fc3 = nn_ms.Dense(84, num_classes)
        self.relu = nn_ms.ReLU()

        # 初始化权重（与PyTorch一致）
        self._initialize_weights()

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn_ms.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    'HeNormal', cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(ms.common.initializer.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn_ms.Dense):
                cell.weight.set_data(ms.common.initializer.initializer(
                    'Normal', cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(ms.common.initializer.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
