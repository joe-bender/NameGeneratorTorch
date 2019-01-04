import torch
import torch.nn as nn

class NamesRNN(nn.Module):
    def __init__(self):
        super(NamesRNN, self).__init__()
        self.LSTM = nn.LSTM(26, 26, 1)

    def forward(self, x, hidden=None):
        y_pred, hidden = self.LSTM(x, hidden)
        return y_pred, hidden
