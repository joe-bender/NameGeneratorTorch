"""Generate names with the network

Once the RNN network has been trained in training.py, it can be used here to
generate names. A first letter for the name is generated randomly and given to
the network as the first input. We take the output logits and convert them to a
letter to add to the name being generated. This is done by passing the logits
through a softmax to get a probability distribution and then sampling from that
distribution to choose a letter. When the network predicts the terminal character
(underscore), the process is over and the previously predicted characters are
concatenated into a name to be output. Each predicted letter is also converted
to a onehot tensor to be fed to the next step of the network along with the
hidden state.
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

    # verify the input is a tensor
    if type(pred) is not torch.Tensor:
        raise Exception('{} is not a tensor'.format(pred))
    # verify the input has the right shape
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

    # convert from shape (1, 1, 27) to shape (27)
    # in other words, get the innermost tensor
    pred = pred.view(-1)
    # convert logits to probabilities
    sm = softmax(pred, dim=0)
    # choose the index with the highest probability
    choice = sm.argmax().item()
    # convert the index to a letter
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

    # convert from shape (1, 1, 27) to shape (27)
    # in other words, get the innermost tensor
    pred = pred.view(-1)
    # convert logits to probabilities, using the tuned softmax function
    sm = softmax_tuned(pred, hps['softmax_tuning'])
    # np.random.choice doesn't think the probabilities add up to 1 unless
    # we convert the torch tensor to a numpy array.
    probs = sm.numpy()
    # create a numpy array of all the categories (or indexes) that can be
    # converted to letters
    categories = np.arange(hps['onehot_length'])
    # use softmax probabilities to sample non-uniformly from all categories
    choice = int(np.random.choice(categories, p=probs))
    # convert the category to a letter
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

    # verify inputs
    assert(type(tuning) in (int, float))
    assert(type(x) is torch.Tensor)
    assert(x.size() == torch.Size([27]))

    # this is the normal softmax function with just the tuning variable added
    x = torch.exp(tuning*x)
    x = x/x.sum()

    # verify output shape
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
        # use the given first letter to start the process
        x = helpers.letter_to_onehot(first_letter)
        # convert to the shape that the LSTM module requires as input
        x = x.view(1, 1, -1)
        # the first hidden input will be zeros since we're starting a new
        # sequence, or new name
        hidden = None

        # loop until ternimal character is predicted
        while True:
            y_pred, hidden = model(x, hidden)
            # here we can choose between deterministic or random conversion
            # from prediction logits to letter
            letter = pred_to_letter_rand(y_pred)
            # stop the process when the terminal character is predicted
            if letter == '_':
                break
            # add this predicted letter to the list of letters
            letters.append(letter)
            # convert predicted letter to a onehot so it can be used as
            # input for the next step (with the required shape)
            x = helpers.letter_to_onehot(letter).view(1, 1, -1)

    # put all the predicted letters together to get a name
    name = ''.join(letters)
    # validate output type
    assert(type(name) is str)
    # make sure we didn't add in some strange characters by accident
    for letter in name:
        assert(letter in string.ascii_lowercase)
    # names should start with a capital letter
    name = name.capitalize()

    return name

# load the model that was trained in taining.py
model = NamesRNN()
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

# generate some names
for _ in range(10):
    first_letter = random.choice(string.ascii_lowercase)
    name = generate_name(first_letter)
    print(name)
