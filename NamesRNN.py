import torch
import torch.nn as nn
from hyperparameters import hps

onehot_length = hps['onehot_length']
num_layers = hps['lstm_layers']

class NamesRNN(nn.Module):
    """Predict the next letter in a name from a sequence of letters

    The inputs include all ascii lowercase letters and the outputs include
    all ascii lowercase letters plus the underscore, which is used as a
    terminal character to mark the end of a name. The network is simply a
    stack of LSTM modules. The inputs must first be converted to onehot tensors,
    while the outputs are in the form of raw logits. 
    """

    def __init__(self):
        super(NamesRNN, self).__init__()
        self.LSTM = nn.LSTM(onehot_length, onehot_length, num_layers)

    def forward(self, x, hidden=None):
        y_pred, hidden = self.LSTM(x, hidden)
        return y_pred, hidden
