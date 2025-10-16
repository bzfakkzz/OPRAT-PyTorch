import numpy as np
import torch.nn as nn
import troubleshooter as ts
import mindspore.nn as nn_ms
import torch
import mindspore as ms

# PyTorch LSTM模型
class TorchLSTMModel0(nn.Module):
    def __init__(self, num_classes=1):
        super(TorchLSTMModel0, self).__init__()
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=50,
                             num_layers=1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=100,
                             num_layers=1, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.5)
        self.dense = nn.Linear(100, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x=x
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        return self.dense(x[:, -1, :])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Dropout):
                # Dropout layers do not have weights to initialize
                pass

# MindSpore LSTM模型
class MindSporeLSTMModel0(nn_ms.Cell):
    def __init__(self, num_classes=1):
        super(MindSporeLSTMModel0, self).__init__()
        self.lstm1 = nn_ms.LSTM(
            input_size=50,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout1 = nn_ms.Dropout(0.5)
        self.lstm2 = nn_ms.LSTM(
            input_size=50,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout2 = nn_ms.Dropout(0.5)
        self.dense = nn_ms.Dense(100, num_classes)

    def construct(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        return self.dense(x[:, -1, :])
