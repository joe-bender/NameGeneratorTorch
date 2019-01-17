import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import helpers
from NamesRNN import NamesRNN
import random
from hyperparameters import hps
import string

def validate_pred_input(pred):
    if type(pred) is not torch.Tensor:
        raise Exception('{} is not a tensor'.format(pred))
    if pred.size() != torch.Size([1, 1, 27]):
        raise Exception('pred must be of size (1, 1, 27)')

def pred_to_letter_det(pred):
    validate_pred_input(pred)

    pred = pred.view(-1)
    sm = softmax(pred, dim=0)
    choice = sm.argmax().item()
    letter = helpers.category_to_letter(choice)
    return letter

def pred_to_letter_rand(pred):
    validate_pred_input(pred)

    pred = pred.view(-1)
    sm = softmax_tuned(pred, hps['softmax_tuning'])
    probs = sm.numpy()
    letters = np.arange(hps['onehot_length'])
    choice = int(np.random.choice(letters, p=probs))
    letter = helpers.category_to_letter(choice)
    return letter

def softmax_tuned(x, tuning):
    assert(type(tuning) in (int, float))
    assert(type(x) is torch.Tensor)
    assert(x.size() == torch.Size([27]))

    x = torch.exp(tuning*x)
    x = x/x.sum()

    assert(x.size() == torch.Size([27]))
    return x

def generate_name(first_letter):
    # validate input
    assert(first_letter in string.ascii_lowercase)
    # keep a list of all the generated letters (starting with the given first letter)
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

    name = ''.join(letters)
    # validate output
    assert(type(name) is str)
    for letter in name:
        assert(letter in string.ascii_lowercase)

    return name.capitalize()

model = NamesRNN()
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

# generate some names
for _ in range(10):
    first_letter = random.choice(string.ascii_lowercase)
    name = generate_name(first_letter)
    print(name)
