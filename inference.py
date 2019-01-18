"""Generate names with the network

Once the RNN network has been trained in training.py, it can be used here to
generate names. A first letter for the name is generated randomly and given to
the network as the first input. We take the first output and the first hidden
state and feed them to the next timestep like in training.py, but we also
take the output logits and convert them to a letter to add to the name
being generated. This is done by passing the logits through a softmax to get
a probability distribution and then sampling from that distribution to choose a
letter. When the network predicts the terminal character (underscore), the
process is over and the previously predicted characters are concatenated into
a name to be output.
"""

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
    """ Validate an input as a prediction tensor

    * pred: the tensor to be validated
    """

    if type(pred) is not torch.Tensor:
        raise Exception('{} is not a tensor'.format(pred))
    if pred.size() != torch.Size([1, 1, 27]):
        raise Exception('pred must be of size (1, 1, 27)')

def pred_to_letter_det(pred):
    """Convert a prediction tensor to a letter deterministically

    Choose the letter based on the tensor dimension with the hightest value.
    This will always result in the same prediction given the same inputs.
    For example, generating a name that starts with 'A' will always give
    the same name.

    * pred: the tensor to be converted
    """

    validate_pred_input(pred)

    pred = pred.view(-1)
    sm = softmax(pred, dim=0)
    choice = sm.argmax().item()
    letter = helpers.category_to_letter(choice)
    return letter

def pred_to_letter_rand(pred):
    """Convert a prediction tensor to a letter randomly

    Choose the letter by sampling from a probability distribution created
    by putting the prediction through a softmax. The probabilities can be
    tuned with the softmax_tuning parameter to make the output more or less
    random.

    * pred: the tensor to be converted
    """

    validate_pred_input(pred)

    pred = pred.view(-1)
    sm = softmax_tuned(pred, hps['softmax_tuning'])
    probs = sm.numpy()
    letters = np.arange(hps['onehot_length'])
    choice = int(np.random.choice(letters, p=probs))
    letter = helpers.category_to_letter(choice)
    return letter

def softmax_tuned(x, tuning):
    """A softmax function that allows for tuning probabilities

    The standard softmax function returns a tensor of probabilities that add
    up to 1. This tuned version amplifies the difference between probabilities,
    so that they will be exaggerated. Putting the tuning at 0 will completely
    smooth out the probabilities so they are all equal. Putting tuning at 1 will
    have no effect. Larger numbers will exaggerate the difference between
    probabilities that may have been close together before the tuning. High
    tuning settings will result in the highest probability being close to 1,
    making pred_to_letter_rand act like pred_to_letter_det.

    * x: the raw prediction tensor (logits)
    * tuning: an int or float used to amplify the probabilities
    """

    assert(type(tuning) in (int, float))
    assert(type(x) is torch.Tensor)
    assert(x.size() == torch.Size([27]))

    x = torch.exp(tuning*x)
    x = x/x.sum()

    assert(x.size() == torch.Size([27]))
    return x

def generate_name(first_letter):
    """Generate a name that starts with the given first letter

    * first_letter: the first letter of the generated name
    """

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

# load the model that was trained in taining.py
model = NamesRNN()
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

# generate some names
for _ in range(10):
    first_letter = random.choice(string.ascii_lowercase)
    name = generate_name(first_letter)
    print(name)
