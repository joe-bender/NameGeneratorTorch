import torch

def letter_to_tensor(letter):
    assert(letter.islower())
    i = ord(letter) - 97
    tensor = torch.zeros(26)
    tensor[i] = 1
    return tensor

def letter_to_category(letter):
    return ord(letter) - 97

def category_to_letter(category):
    return chr(category + 97)

def tensor_to_letter(tensor):
    return category_to_letter(tensor.argmax().item())
