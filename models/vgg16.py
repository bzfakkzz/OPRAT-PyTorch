import torch
import torch.nn as nn
import mindspore as ms
import mindspore.nn as nn_ms
import troubleshooter as ts
import os

device=torch.device('cuda')

class TorchVGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(TorchVGG16, self).__init__()
        # Block1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1).cuda(device)
        self.relu1_1 = nn.ReLU().cuda(device)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1).cuda(device)
        self.relu1_2 = nn.ReLU().cuda(device)
        self.maxpool1 = nn.MaxPool2d(2, 2).cuda(device)

        # Block2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1).cuda(device)
        self.relu2_1 = nn.ReLU().cuda(device)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1).cuda(device)
        self.relu2_2 = nn.ReLU().cuda(device)
        self.maxpool2 = nn.MaxPool2d(2, 2).cuda(device)

        # Block3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1).cuda(device)
        self.relu3_1 = nn.ReLU().cuda(device)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1).cuda(device)
        self.relu3_2 = nn.ReLU().cuda(device)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1).cuda(device)
        self.relu3_3 = nn.ReLU().cuda(device)
        self.maxpool3 = nn.MaxPool2d(2, 2).cuda(device)

        # Block4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1).cuda(device)
        self.relu4_1 = nn.ReLU().cuda(device)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1).cuda(device)
        self.relu4_2 = nn.ReLU().cuda(device)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1).cuda(device)
        self.relu4_3 = nn.ReLU().cuda(device)
        self.maxpool4 = nn.MaxPool2d(2, 2).cuda(device)

        # Block5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1).cuda(device)
        self.relu5_1 = nn.ReLU().cuda(device)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1).cuda(device)
        self.relu5_2 = nn.ReLU().cuda(device)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1).cuda(device)
        self.relu5_3 = nn.ReLU().cuda(device)
        self.maxpool5 = nn.MaxPool2d(2, 2).cuda(device)

        # Classifier
        self.flatten = nn.Flatten().cuda(device)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096).cuda(device)
        self.relu_fc1 = nn.ReLU().cuda(device)
        self.dropout1 = nn.Dropout(0.5).cuda(device)
        self.fc2 = nn.Linear(4096, 4096).cuda(device)
        self.relu_fc2 = nn.ReLU().cuda(device)
        self.dropout2 = nn.Dropout(0.5).cuda(device)
        self.fc3 = nn.Linear(4096, num_classes).cuda(device)
        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x=x.cuda(device)
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.maxpool1(x)
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.maxpool2(x)
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.maxpool3(x)
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.maxpool4(x)
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

class MindSporeVGG16(nn_ms.Cell):
    def __init__(self, num_classes=100):
        super(MindSporeVGG16, self).__init__()
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
        self.maxpool3 = nn_ms.MaxPool2d(2, 2)

        # Block4
        self.conv4_1 = nn_ms.Conv2d(256, 512, 3, padding=1, pad_mode='pad')
        self.relu4_1 = nn_ms.ReLU()
        self.conv4_2 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu4_2 = nn_ms.ReLU()
        self.conv4_3 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu4_3 = nn_ms.ReLU()
        self.maxpool4 = nn_ms.MaxPool2d(2, 2)

        # Block5
        self.conv5_1 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu5_1 = nn_ms.ReLU()
        self.conv5_2 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu5_2 = nn_ms.ReLU()
        self.conv5_3 = nn_ms.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.relu5_3 = nn_ms.ReLU()
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
        # 保持原结构不变
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.maxpool1(x)
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.maxpool2(x)
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.maxpool3(x)
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.maxpool4(x)
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
