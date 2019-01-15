import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import helpers
from NamesRNN import NamesRNN
from random import randint
from hyperparameters import hps

def pred_to_letter_det(pred):
    pred = pred.view(-1)
    sm = softmax(pred, dim=0)
    choice = sm.argmax().item()
    letter = helpers.category_to_letter(choice)
    return letter

def pred_to_letter_rand(pred):
    pred = pred.view(-1)
    sm = softmax_tuned(pred, hps['softmax_tuning'])
    probs = sm.numpy()
    letters = np.arange(hps['onehot_length'])
    choice = np.random.choice(letters, p=probs)
    letter = helpers.category_to_letter(choice)
    return letter

def softmax_tuned(x, tuning):
    x = torch.exp(tuning*x)
    x = x/x.sum()
    return x

def generate_name(first_letter):
    letters = [first_letter]
    with torch.no_grad():
        x = helpers.letter_to_onehot(first_letter)
        x = x.view(1, 1, -1)
        hidden = None

        while True:
            y_pred, hidden = model(x, hidden)
            letter = pred_to_letter_rand(y_pred)
            if letter == '_':
                break
            letters.append(letter)
            x = helpers.letter_to_onehot(letter).view(1, 1, -1)

    return ''.join(letters).capitalize()

model = NamesRNN()
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

first_letter = helpers.category_to_letter(randint(0, hps['onehot_length'] - 2))
name = generate_name(first_letter)
print(name)
