import torch
import pandas as pd

def letter_to_tensor(letter):
    assert(letter.islower())
    i = ord(letter) - 97
    tensor = torch.zeros(26)
    tensor[i] = 1
    return tensor

def letter_to_category(letter):
    return ord(letter) - 97

def get_words(filename):
    data = pd.read_csv(filename, header=None, index_col=False)
    names = data.iloc[:, 0]
    names = list(names)
    names = [name.lower() for name in names if len(name) == 6]
    return names

def get_data(filename):
    words = get_words(filename)

    # inputs include all letters except the last
    x = []
    for letter in range(5):
        x.append(torch.stack([letter_to_tensor(word[letter]) for word in words]))
    x = torch.stack(x)

    # targets include all letters except the first
    y = []
    for letter in range(1, 6):
        y.append([letter_to_category(word[letter]) for word in words])
    y = torch.tensor(y)

    return x, y

# words = get_words('names2017.csv')
# print(len(words))
