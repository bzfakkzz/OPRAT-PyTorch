import numpy as np
import torch
import torch.nn as nn
import mindspore as ms
from mindspore import context
import troubleshooter as ts
import mindspore.nn as nn_ms

device=torch.device('cuda')

# ---------------------------- DepthwiseSeparableConv2d 定义 ----------------------------
class MindSporeDepthwiseSeparableConv2d(nn_ms.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MindSporeDepthwiseSeparableConv2d, self).__init__()
        padding = (kernel_size - 1) // 2

        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn_ms.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pad_mode='pad',
            group=in_channels,
            has_bias=False
        )
        self.bn1 = nn_ms.BatchNorm2d(in_channels)

        # 点卷积 (Pointwise Convolution)
        self.pointwise = nn_ms.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            has_bias=False
        )
        self.bn2 = nn_ms.BatchNorm2d(out_channels)
        self.relu = nn_ms.ReLU()

    def construct(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# ---------------------------- 1. PyTorch MobileNet ----------------------------
class TorchDepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(TorchDepthwiseSeparableConv2d, self).__init__()
        padding = (kernel_size - 1) // 2

        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        ).to('cuda')
        self.bn1 = nn.BatchNorm2d(in_channels).to('cuda')

        # 点卷积 (Pointwise Convolution)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        ).to('cuda')
        self.bn2 = nn.BatchNorm2d(out_channels).to('cuda')
        self.relu = nn.ReLU().to('cuda')

    def forward(self, x):
        x=x.to('cuda')
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class TorchMobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(TorchMobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1).to('cuda')
        self.bn1 = nn.BatchNorm2d(32).to('cuda')
        self.relu = nn.ReLU().to('cuda')

        # Depthwise separable convolutions
        self.conv_dw1 = TorchDepthwiseSeparableConv2d(32, 64, kernel_size=3).to('cuda')
        self.conv_dw2 = TorchDepthwiseSeparableConv2d(64, 128, kernel_size=3, stride=2).to('cuda')
        self.conv_dw3 = TorchDepthwiseSeparableConv2d(128, 128, kernel_size=3).to('cuda')
        self.conv_dw4 = TorchDepthwiseSeparableConv2d(128, 256, kernel_size=3, stride=2).to('cuda')
        self.conv_dw5 = TorchDepthwiseSeparableConv2d(256, 256, kernel_size=3).to('cuda')
        self.conv_dw6 = TorchDepthwiseSeparableConv2d(256, 512, kernel_size=3, stride=2).to('cuda')

        # Repeat this block 5 times
        self.conv_dw_repeated = nn.Sequential(
            *[TorchDepthwiseSeparableConv2d(512, 512, kernel_size=3) for _ in range(5)]
        ).to('cuda')

        self.conv_dw7 = TorchDepthwiseSeparableConv2d(512, 1024, kernel_size=3, stride=2).to('cuda')
        self.conv_dw8 = TorchDepthwiseSeparableConv2d(1024, 1024, kernel_size=3).to('cuda')

        self.avg_pool = nn.AdaptiveAvgPool2d(1).to('cuda')
        self.flatten = nn.Flatten().to('cuda')
        self.fc = nn.Linear(1024, num_classes).to('cuda')

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x=x.to('cuda')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)

        x = self.conv_dw_repeated(x)

        x = self.conv_dw7(x)
        x = self.conv_dw8(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ---------------------------- 2. MindSpore MobileNet ----------------------------
class MindSporeMobileNet(nn_ms.Cell):
    def __init__(self, num_classes=1000):
        super(MindSporeMobileNet, self).__init__()
        self.conv1 = nn_ms.Conv2d(3, 32, kernel_size=3, stride=2, pad_mode='same')
        self.bn1 = nn_ms.BatchNorm2d(32)
        self.relu = nn_ms.ReLU()

        # Depthwise separable convolutions
        self.conv_dw1 = MindSporeDepthwiseSeparableConv2d(32, 64, kernel_size=3)
        self.conv_dw2 = MindSporeDepthwiseSeparableConv2d(64, 128, kernel_size=3, stride=2)
        self.conv_dw3 = MindSporeDepthwiseSeparableConv2d(128, 128, kernel_size=3)
        self.conv_dw4 = MindSporeDepthwiseSeparableConv2d(128, 256, kernel_size=3, stride=2)
        self.conv_dw5 = MindSporeDepthwiseSeparableConv2d(256, 256, kernel_size=3)
        self.conv_dw6 = MindSporeDepthwiseSeparableConv2d(256, 512, kernel_size=3, stride=2)

        # Repeat this block 5 times
        self.conv_dw_repeated = nn_ms.SequentialCell(
            [MindSporeDepthwiseSeparableConv2d(512, 512, kernel_size=3) for _ in range(5)]
        )

        self.conv_dw7 = MindSporeDepthwiseSeparableConv2d(512, 1024, kernel_size=3, stride=2)
        self.conv_dw8 = MindSporeDepthwiseSeparableConv2d(1024, 1024, kernel_size=3)

        # 注意：输入尺寸为224x224时，最终特征图尺寸为7x7
        self.avg_pool = nn_ms.AvgPool2d(kernel_size=7)
        self.flatten = nn_ms.Flatten()
        self.fc = nn_ms.Dense(1024, num_classes)

        # 初始化权重
        self._initialize_weights()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)

        x = self.conv_dw_repeated(x)

        x = self.conv_dw7(x)
        x = self.conv_dw8(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn_ms.Conv2d, nn_ms.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    'HeNormal', cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(ms.common.initializer.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn_ms.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer(
                    'ones', cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer(
                    'zeros', cell.beta.shape, cell.beta.dtype))
