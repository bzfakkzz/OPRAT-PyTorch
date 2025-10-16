import numpy as np
import torch.nn as nn
import troubleshooter as ts
import mindspore.nn as nn_ms
import torch
import mindspore as ms
from mindspore import context

# ---------------------------- del. PyTorch VGG19 ----------------------------
class TorchVGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(TorchVGG19, self).__init__()
        # Block1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # Block2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # Block3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)  # VGG19增加的层
        self.relu3_4 = nn.ReLU()  # VGG19增加的层
        self.maxpool3 = nn.MaxPool2d(2, 2)

        # Block4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)  # VGG19增加的层
        self.relu4_4 = nn.ReLU()  # VGG19增加的层
        self.maxpool4 = nn.MaxPool2d(2, 2)

        # Block5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)  # VGG19增加的层
        self.relu5_4 = nn.ReLU()  # VGG19增加的层
        self.maxpool5 = nn.MaxPool2d(2, 2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x=x
        # Block1
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.maxpool1(x)

        # Block2
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.maxpool2(x)

        # Block3
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.relu3_4(self.conv3_4(x))  # VGG19增加的层
        x = self.maxpool3(x)

        # Block4
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.relu4_4(self.conv4_4(x))  # VGG19增加的层
        x = self.maxpool4(x)

        # Block5
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.relu5_4(self.conv5_4(x))  # VGG19增加的层
        x = self.maxpool5(x)

        # Classifier
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ---------------------------- 2. MindSpore VGG19 ----------------------------
class MindSporeVGG19(nn_ms.Cell):
    def __init__(self, num_classes=1000):
        super(MindSporeVGG19, self).__init__()
        # Block1
        self.conv1_1 = nn_ms.Conv2d(3, 64, 3, padding=1, pad_mode='pad')
        self.relu1_1 = nn_ms.ReLU()
        self.conv1_2 = nn_ms.Conv2d(64, 64, 3, padding=1, pad_mode='pad')
        self.relu1_2 = nn_ms.ReLU()
        self.maxpool1 = nn_ms.MaxPool2d(2, 2)

        # Block2
        self.conv2_1 = nn_ms.Conv2d(64, 128, 3, padding=1, pad_mode='pad')
        self.relu2_1 = nn_ms.ReLU()
        self.conv2_2 = nn_ms.Conv2d(128, 128, 3, padding=1, pad_mode='pad')
        self.relu2_2 = nn_ms.ReLU()
        self.maxpool2 = nn_ms.MaxPool2d(2, 2)

        # Block3
        self.conv3_1 = nn_ms.Conv2d(128, 256, 3, padding=1, pad_mode='pad')
        self.relu3_1 = nn_ms.ReLU()
        self.conv3_2 = nn_ms.Conv2d(256, 256, 3, padding=1, pad_mode='pad')
        self.relu3_2 = nn_ms.ReLU()
        self.conv3_3 = nn_ms.Conv2d(256, 256, 3, padding=1, pad_mode='pad')
        self.relu3_3 = nn_ms.ReLU()
        self.conv3_4 = nn_ms.Conv2d(256, 256, 3, padding=1, pad_mode='pad')  # VGG19增加的层
        self.relu3_4 = nn_ms.ReLU()  # VGG19增加的层
        self.maxpool3 = nn_ms.MaxPool2d(2, 2)

        # Block4
        self.conv4_1 = nn_ms.Conv2d(256, 512, 3, padding=1, pad_mode='pad')
        self.relu4_1 = nn_ms.ReLU()
        self.conv4_2 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu4_2 = nn_ms.ReLU()
        self.conv4_3 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu4_3 = nn_ms.ReLU()
        self.conv4_4 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')  # VGG19增加的层
        self.relu4_4 = nn_ms.ReLU()  # VGG19增加的层
        self.maxpool4 = nn_ms.MaxPool2d(2, 2)

        # Block5
        self.conv5_1 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu5_1 = nn_ms.ReLU()
        self.conv5_2 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu5_2 = nn_ms.ReLU()
        self.conv5_3 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu5_3 = nn_ms.ReLU()
        self.conv5_4 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')  # VGG19增加的层
        self.relu5_4 = nn_ms.ReLU()  # VGG19增加的层
        self.maxpool5 = nn_ms.MaxPool2d(2, 2)

        # Classifier
        self.flatten = nn_ms.Flatten()
        self.fc1 = nn_ms.Dense(512 * 7 * 7, 4096)
        self.relu_fc1 = nn_ms.ReLU()
        self.dropout1 = nn_ms.Dropout(0.5)
        self.fc2 = nn_ms.Dense(4096, 4096)
        self.relu_fc2 = nn_ms.ReLU()
        self.dropout2 = nn_ms.Dropout(0.5)
        self.fc3 = nn_ms.Dense(4096, num_classes)

    def construct(self, x):
        # Block1
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.maxpool1(x)

        # Block2
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.maxpool2(x)

        # Block3
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.relu3_4(self.conv3_4(x))  # VGG19增加的层
        x = self.maxpool3(x)

        # Block4
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.relu4_4(self.conv4_4(x))  # VGG19增加的层
        x = self.maxpool4(x)

        # Block5
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.relu5_4(self.conv5_4(x))  # VGG19增加的层
        x = self.maxpool5(x)

        # Classifier
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
