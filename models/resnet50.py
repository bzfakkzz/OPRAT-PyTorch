import numpy as np
import torch
import torch.nn as nn
import mindspore as ms
from mindspore import context
import troubleshooter as ts
import mindspore.nn as nn_ms
from mindspore.common.initializer import Normal, initializer

# ---------------------------- del. PyTorch ResNet-50 ----------------------------
class TorchBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(TorchBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        x=x
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class TorchResNet50(nn.Module):
    def __init__(self, block=TorchBottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        super(TorchResNet50, self).__init__()
        self.inplanes = 64
        # stem
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 4 stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x=x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ---------------------------- 2. MindSpore ResNet-50 ----------------------------
class MindSporeBottleneck(nn_ms.Cell):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(MindSporeBottleneck, self).__init__()
        self.conv1 = nn_ms.Conv2d(in_channel, out_channel, 1, has_bias=False)
        self.bn1 = nn_ms.BatchNorm2d(out_channel)
        self.conv2 = nn_ms.Conv2d(out_channel, out_channel, 3, stride=stride,
                                  pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn_ms.BatchNorm2d(out_channel)
        self.conv3 = nn_ms.Conv2d(out_channel, out_channel * self.expansion,
                                  1, has_bias=False)
        self.bn3 = nn_ms.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn_ms.ReLU()
        self.downsample = downsample

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class MindSporeResNet50(nn_ms.Cell):
    def __init__(self, block=MindSporeBottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        super(MindSporeResNet50, self).__init__()
        self.in_channel = 64

        # stem
        self.conv1 = nn_ms.Conv2d(3, 64, 7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn_ms.BatchNorm2d(64)
        self.relu = nn_ms.ReLU()
        self.maxpool = nn_ms.MaxPool2d(3, stride=2, pad_mode='pad')

        # 4 stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # head
        self.avgpool = nn_ms.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn_ms.Flatten()
        self.fc = nn_ms.Dense(512 * block.expansion, num_classes)

        # 初始化权重
        self._init_weights()

    def _make_layer(self, block, channel, block_num, stride):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn_ms.SequentialCell([
                nn_ms.Conv2d(self.in_channel, channel * block.expansion,
                             1, stride=stride, has_bias=False),
                nn_ms.BatchNorm2d(channel * block.expansion)
            ])

        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, stride=1, downsample=None))
        return nn_ms.SequentialCell(layers)

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn_ms.Conv2d):
                cell.weight.set_data(initializer(Normal(0.0, 0.02), cell.weight.shape))
            elif isinstance(cell, nn_ms.BatchNorm2d):
                cell.gamma.set_data(initializer('ones', cell.gamma.shape))
                cell.beta.set_data(initializer('zeros', cell.beta.shape))
            elif isinstance(cell, nn_ms.Dense):
                cell.weight.set_data(initializer(Normal(0.0, 0.02), cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape))

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
