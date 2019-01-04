import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
from helpers import letter_to_tensor, letter_to_category, category_to_letter, tensor_to_letter

model = nn.LSTM(26, 26, 2)
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

def pred_to_letter_det(pred):
    pred = pred.view(-1)
    sm = softmax(pred, dim=0)
    choice = sm.argmax().item()
    letter = category_to_letter(choice)
    return letter

def pred_to_letter_rand(pred):
    pred = pred.view(-1)
    sm = softmax_tuned(pred, 10)
    probs = sm.numpy()
    letters = np.arange(26)
    choice = np.random.choice(letters, p=probs)
    letter = category_to_letter(choice)
    return letter

def softmax_tuned(x, tuning):
    x = torch.exp(tuning*x)
    x = x/x.sum()
    print(x.max())
    return x

def generate_name(first_letter):
    letters = [first_letter]
    with torch.no_grad():
        x = letter_to_tensor(first_letter)
        x = x.view(1, 1, -1)
        hidden = (torch.zeros(2, 1, 26), torch.zeros(2, 1, 26))

        for _ in range(5):
            y_pred, hidden = model(x, hidden)
            letter = pred_to_letter_rand(y_pred)
            letters.append(letter)
            x = letter_to_tensor(letter).view(1, 1, -1)

    return ''.join(letters).capitalize()

letter = category_to_letter(torch.LongTensor(1).random_(26).item())
name = generate_name(letter)
print(name)
