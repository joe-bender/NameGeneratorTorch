import torch

char_to_i = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23,
    'y': 24,
    'z': 25,
    '_': 26,
}
i_to_char = {i: char for char, i in char_to_i.items()}
onehot_length = len(char_to_i)

def letter_to_onehot(letter):
    i = char_to_i[letter]
    onehot = torch.zeros(onehot_length)
    onehot[i] = 1
    return onehot.view(1, 1, -1)

def letter_to_category(letter):
    return torch.tensor([char_to_i[letter]])

def category_to_letter(category):
    return i_to_char[category]

def onehot_to_letter(onehot):
    return category_to_letter(onehot.argmax().item())

def name_to_xy(name):
    # the inputs are the lowercase letters of the name
    xs = list(name.lower())
    # the outputs are the inputs shifted over by one, plus a terminal character
    ys = xs[1:]+['_']

    xs = [letter_to_onehot(x) for x in xs]
    ys = [letter_to_category(y) for y in ys]

    return xs, ys
