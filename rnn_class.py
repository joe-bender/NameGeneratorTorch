import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, D_in, D_out, layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(D_in, D_out, layers)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
