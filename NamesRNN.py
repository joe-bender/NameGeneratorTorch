"""The network that will be used in the training and inference steps"""

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
        """Set input and output dimensions and number of layers"""

        super(NamesRNN, self).__init__()
        self.LSTM = nn.LSTM(onehot_length, onehot_length, num_layers)

    def forward(self, x, hidden=None):
        """Send the input and hidden layers into the LSTM stack

        * x: input in the form of a sequence of batches of onehot tensors
        representing letters.
        * hidden: a tuple of the C and h tensors to be passed sideways into the
        next timestep of the LSTM. The default is None so that the first step of
        the sequence is given tensors of zeros.
        """

        y_pred, hidden = self.LSTM(x, hidden)
        return y_pred, hidden
