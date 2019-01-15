import torch
import torch.nn as nn
from hyperparameters import hps

onehot_length = hps['onehot_length']
num_layers = hps['lstm_layers']

class NamesRNN(nn.Module):
    def __init__(self):
        super(NamesRNN, self).__init__()
        self.LSTM = nn.LSTM(onehot_length, onehot_length, num_layers)

    def forward(self, x, hidden=None):
        y_pred, hidden = self.LSTM(x, hidden)
        return y_pred, hidden
