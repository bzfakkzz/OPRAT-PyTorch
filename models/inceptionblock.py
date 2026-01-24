import numpy as np
import torch
import torch.nn as nn
import troubleshooter as ts

# ---------------------------- del. PyTorch InceptionBlock ----------------------------
class TorchInceptionBlock(nn.Module):
    def __init__(self, in_channels, c1=16, c2=(32,64), c3=(16,32), c4=16):
        super(TorchInceptionBlock, self).__init__()
        # Path del
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1, bias=True)

        # Path 2
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1, bias=True)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1, bias=True)

        # Path 3
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1, bias=True)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2, bias=True)

        # Path 4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1, bias=True)

        self.batch_norm = nn.BatchNorm2d(c1 + c2[1] + c3[1] + c4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=x
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        concatenated = torch.cat([p1, p2, p3, p4], dim=1)
        normalized = self.batch_norm(concatenated)
        return self.relu(normalized)
