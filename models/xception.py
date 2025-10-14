import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mindspore as ms
from mindspore import context, ops
import troubleshooter as ts
from mindspore.common.initializer import HeNormal
import mindspore.nn as nn_ms

device=torch.device('cuda')

# ---------------------------- 1. PyTorch Xception ----------------------------
class TorchSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(TorchSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False).to('cuda')
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False).to('cuda')

        # 初始化权重
        self._init_weights()

    def forward(self, x):
        x=x.to('cuda')
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    def _init_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

class TorchXceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, start_with_relu=True, grow_first=True):
        super(TorchXceptionBlock, self).__init__()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False).to('cuda')
        self.residual_bn = nn.BatchNorm2d(out_channels).to('cuda')

        self.sep_conv1 = TorchSeparableConv2d(in_channels if grow_first else out_channels,
                                              out_channels, 3, padding=1).to('cuda')
        self.bn1 = nn.BatchNorm2d(out_channels).to('cuda')
        self.sep_conv2 = TorchSeparableConv2d(out_channels, out_channels, 3, padding=1).to('cuda')
        self.bn2 = nn.BatchNorm2d(out_channels).to('cuda')
        self.sep_conv3 = TorchSeparableConv2d(out_channels, out_channels, 3, padding=1).to('cuda')
        self.bn3 = nn.BatchNorm2d(out_channels).to('cuda')
        self.maxpool = nn.MaxPool2d(3, strides, padding=1).to('cuda')

        self.start_with_relu = start_with_relu.to('cuda')
        self.grow_first = grow_first.to('cuda')
        self.relu = nn.ReLU().to('cuda')

        # 初始化权重
        self._init_weights()

    def forward(self, x):
        x=x.to('cuda')
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)

        if self.start_with_relu:
            x = F.relu(x)
        if self.grow_first:
            x = self.sep_conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
        x = self.sep_conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if not self.grow_first:
            x = self.sep_conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.sep_conv3(x)
        x = self.bn3(x)

        x = self.maxpool(x)

        x += residual
        return x

    def _init_weights(self):
        nn.init.kaiming_normal_(self.residual_conv.weight, mode='fan_out', nonlinearity='relu')

class TorchXception(nn.Module):
    def __init__(self, num_classes=1000):
        super(TorchXception, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=False).to('cuda')
        self.bn1 = nn.BatchNorm2d(32).to('cuda')

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False).to('cuda')
        self.bn2 = nn.BatchNorm2d(64).to('cuda')

        # Entry flow
        self.block1 = TorchXceptionBlock(64, 128, strides=2, start_with_relu=False, grow_first=True).to('cuda')
        self.block2 = TorchXceptionBlock(128, 256, strides=2, start_with_relu=True, grow_first=True).to('cuda')
        self.block3 = TorchXceptionBlock(256, 728, strides=2, start_with_relu=True, grow_first=True).to('cuda')

        # Middle flow
        self.mid_blocks = nn.ModuleList().to('cuda')
        for _ in range(8):
            self.mid_blocks.append(TorchXceptionBlock(728, 728, strides=1, start_with_relu=True, grow_first=True))

        # Exit flow
        self.block4 = TorchXceptionBlock(728, 1024, strides=2, start_with_relu=True, grow_first=True).to('cuda')
        self.sep_conv_last = TorchSeparableConv2d(1024, 1536, 3, padding=1).to('cuda')
        self.bn_last1 = nn.BatchNorm2d(1536).to('cuda')
        self.sep_conv_last2 = TorchSeparableConv2d(1536, 2048, 3, padding=1).to('cuda')
        self.bn_last2 = nn.BatchNorm2d(2048).to('cuda')

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to('cuda')
        self.fc = nn.Linear(2048, num_classes).to('cuda')

        # 初始化权重
        self._init_weights()

    def forward(self, x):
        x=x.to('cuda')
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        for block in self.mid_blocks:
            x = block(x)

        x = self.block4(x)
        x = self.sep_conv_last(x)
        x = self.bn_last1(x)
        x = F.relu(x)
        x = self.sep_conv_last2(x)
        x = self.bn_last2(x)
        x = F.relu(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc.weight, 0, 0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

# ---------------------------- 2. MindSpore Xception ----------------------------
class MindSporeSeparableConv2d(nn_ms.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(MindSporeSeparableConv2d, self).__init__()
        self.depthwise = nn_ms.Conv2d(in_channels, in_channels, kernel_size, stride,
                                      pad_mode='pad', padding=padding, group=in_channels,
                                      has_bias=False, weight_init=HeNormal())
        self.pointwise = nn_ms.Conv2d(in_channels, out_channels, 1, 1, pad_mode='pad',
                                      padding=0, has_bias=False, weight_init=HeNormal())

    def construct(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MindSporeXceptionBlock(nn_ms.Cell):
    def __init__(self, in_channels, out_channels, strides=1, start_with_relu=True, grow_first=True):
        super(MindSporeXceptionBlock, self).__init__()
        self.residual_conv = nn_ms.Conv2d(in_channels, out_channels, 1, strides,
                                          has_bias=False, pad_mode='pad', padding=0, weight_init=HeNormal())
        self.residual_bn = nn_ms.BatchNorm2d(out_channels)

        self.sep_conv1 = MindSporeSeparableConv2d(in_channels if grow_first else out_channels,
                                                  out_channels, 3, padding=1)
        self.bn1 = nn_ms.BatchNorm2d(out_channels)
        self.sep_conv2 = MindSporeSeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn_ms.BatchNorm2d(out_channels)
        self.sep_conv3 = MindSporeSeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn_ms.BatchNorm2d(out_channels)
        self.maxpool = nn_ms.MaxPool2d(3, strides, pad_mode='same')

        self.start_with_relu = start_with_relu
        self.grow_first = grow_first
        self.relu = ops.ReLU()

        # 初始化权重
        self._init_weights()

    def construct(self, x):
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)

        if self.start_with_relu:
            x = ops.ReLU()(x)
        if self.grow_first:
            x = self.sep_conv1(x)
            x = self.bn1(x)
            x = ops.ReLU()(x)
        x = self.sep_conv2(x)
        x = self.bn2(x)
        x = ops.ReLU()(x)

        if not self.grow_first:
            x = self.sep_conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.sep_conv3(x)
        x = self.bn3(x)

        x = self.maxpool(x)

        x += residual
        return x

    def _init_weights(self):
        # MindSpore的权重初始化在构造函数中通过weight_init参数完成
        pass

class MindSporeXception(nn_ms.Cell):
    def __init__(self, num_classes=1000):
        super(MindSporeXception, self).__init__()
        self.conv1 = nn_ms.Conv2d(3, 32, 3, stride=2, padding=0, pad_mode='valid',
                                  has_bias=False, weight_init=HeNormal())
        self.bn1 = nn_ms.BatchNorm2d(32)

        self.conv2 = nn_ms.Conv2d(32, 64, 3, padding=1, pad_mode='pad',
                                  has_bias=False, weight_init=HeNormal())
        self.bn2 = nn_ms.BatchNorm2d(64)

        # Entry flow
        self.block1 = MindSporeXceptionBlock(64, 128, strides=2, start_with_relu=False, grow_first=True)
        self.block2 = MindSporeXceptionBlock(128, 256, strides=2, start_with_relu=True, grow_first=True)
        self.block3 = MindSporeXceptionBlock(256, 728, strides=2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.mid_blocks = nn_ms.CellList()
        for _ in range(8):
            self.mid_blocks.append(MindSporeXceptionBlock(728, 728, strides=1, start_with_relu=True, grow_first=True))

        # Exit flow
        self.block4 = MindSporeXceptionBlock(728, 1024, strides=2, start_with_relu=True, grow_first=True)
        self.sep_conv_last = MindSporeSeparableConv2d(1024, 1536, 3, padding=1)
        self.bn_last1 = nn_ms.BatchNorm2d(1536)
        self.sep_conv_last2 = MindSporeSeparableConv2d(1536, 2048, 3, padding=1)
        self.bn_last2 = nn_ms.BatchNorm2d(2048)

        # 使用自适应平均池化替代固定尺寸池化
        self.avg_pool = nn_ms.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn_ms.Flatten()
        self.fc = nn_ms.Dense(2048, num_classes, weight_init=HeNormal())

        # 初始化权重
        self._init_weights()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = ops.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = ops.ReLU()(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        for i in range(8):
            x = self.mid_blocks[i](x)

        x = self.block4(x)
        x = self.sep_conv_last(x)
        x = self.bn_last1(x)
        x = ops.ReLU()(x)
        x = self.sep_conv_last2(x)
        x = self.bn_last2(x)
        x = ops.ReLU()(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _init_weights(self):
        # 批归一化层初始化
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn_ms.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer(
                    'ones', cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer(
                    'zeros', cell.beta.shape, cell.beta.dtype))