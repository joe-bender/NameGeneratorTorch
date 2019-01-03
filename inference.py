import torch
import torch.nn as nn
from torch.nn.functional import softmax
from get_data import letter_to_tensor, letter_to_category

def category_to_letter(cat):
    return chr(cat + 97)

class RNN(nn.Module):
    def __init__(self, D_in, D_out, layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(D_in, D_out, layers)

    def forward(self, x, hidden_in):
        output, hidden_out = self.lstm(x, hidden_in)
        return output, hidden_out

model = RNN(26, 26, 2)
model.eval()

with torch.no_grad():
    letter = 'a'
    x = letter_to_tensor(letter)
    x = x.view(1, 1, -1)
    print(x)

    y_pred = model(x)
    softmax = softmax(y_pred, dim=2)
    print(softmax)
    choice = softmax.argmax().item()
    print(choice)
    print(category_to_letter(choice))
