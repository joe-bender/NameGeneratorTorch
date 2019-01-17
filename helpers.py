import torch
import string

# create dictionaries to convert between characters and ints
chars = list(string.ascii_lowercase+'_')
ints = range(27)
char_to_i = {char: i for char, i in zip(chars, ints)}
i_to_char = {i: char for char, i in char_to_i.items()}
onehot_length = len(char_to_i)

def validate_letter_input(letter):
    """Make sure the input letter is actually a letter

    * letter: the letter to validate
    """

    # verify the input is of type string
    if type(letter) is not str:
        raise Exception('{} is not a string'.format(letter))
    # verify the input is one of the characters that the RNN can process
    if letter not in char_to_i.keys():
        raise Exception('{} is not a lowercase letter or the underscore'.format(letter))

def letter_to_onehot(letter):
    """Convert a letter to a onehot tensor

    * letter: the letter to convert
    """

    validate_letter_input(letter)
    i = char_to_i[letter]
    onehot = torch.zeros(onehot_length)
    onehot[i] = 1
    return onehot.view(1, 1, -1)

def letter_to_category(letter):
    """Convert a letter to a classification category

    * letter: the letter to convert
    """

    validate_letter_input(letter)
    return torch.tensor([char_to_i[letter]])

def category_to_letter(category):
    """Convert a classification category to a letter

    * category: the category to convert
    """

    # verify the input is of type int
    if type(category) is not int:
        raise Exception('{} is not an int'.format(category))
    # verify the input is in the range (0, 26)
    if category not in range(27):
        raise Exception('{} is not in the range (0, 26)'.format(category))

    return i_to_char[category]

def name_to_xy(name):
    """Convert a name to a sequence of training data for the network

    * name: the name to convert
    """
    
    # verify the input is of type string
    if type(name) is not str:
        raise Exception('{} is not a string'.format(name))
    # verify that each character of name is a letter
    for letter in name:
        if letter not in string.ascii_letters:
            raise Exception('{} is not a letter'.format(letter))

    # the inputs are the lowercase letters of the name
    xs = list(name.lower())
    # the outputs are the inputs shifted over by one, plus a terminal character
    ys = xs[1:]+['_']

    xs = [letter_to_onehot(x) for x in xs]
    ys = [letter_to_category(y) for y in ys]

    return xs, ys
