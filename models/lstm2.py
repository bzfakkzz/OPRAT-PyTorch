import numpy as np
import torch
import torch.nn as nn
import mindspore
from mindspore import context
import troubleshooter as ts
import mindspore.nn as nn_ms
from mindspore.common.initializer import Normal

device=torch.device('cuda')

# ---------------------------- 1. PyTorch LSTMModel2 ----------------------------
class TorchLSTMModel2(nn.Module):
    def __init__(self,output_size, input_size=10, hidden_size=32, dropout_rate=0.2):
        super(TorchLSTMModel2, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True).to('cuda')
        self.dropout = nn.Dropout(dropout_rate).to('cuda')
        self.dense = nn.Linear(hidden_size, output_size).to('cuda')

    def forward(self, x):
        x=x.to('cuda')
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # 取最后一个时间步的输出
        output = self.dropout(output)
        output = self.dense(output)
        return output

# ---------------------------- 2. MindSpore LSTMModel2 ----------------------------
class MindSporeLSTMModel2(nn_ms.Cell):
    def __init__(self, output_size, input_size=10, hidden_size=32, dropout_rate=0.2):
        super(MindSporeLSTMModel2, self).__init__()
        self.lstm = nn_ms.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn_ms.Dropout(keep_prob=1 - dropout_rate)
        self.dense = nn_ms.Dense(hidden_size, output_size, weight_init=Normal(0.02), has_bias=True)

    def construct(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # 取最后一个时间步的输出
        output = self.dropout(output)
        output = self.dense(output)
        return output
