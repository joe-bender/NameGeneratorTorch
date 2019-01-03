import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from helpers import letter_to_tensor, letter_to_category, category_to_letter, tensor_to_letter

class WordsDataset(Dataset):
    def __init__(self, filename):
        self.x, self.y = get_data(filename)
        self.seq_length = self.x.shape[0]
        self.features = self.x.shape[2]

    def __len__(self):
        assert self.x.shape[1] == self.y.shape[1]
        return self.x.shape[1]

    def __getitem__(self, i):
        return self.x[:, i], self.y[:, i]

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

# keep structure intact of (sequence, batch, features)
def collate(data):
    x = [tup[0] for tup in data]
    x = torch.stack(x, dim=1)
    y = [tup[1] for tup in data]
    y = torch.stack(y, dim=1)
    return x, y

def get_dataloader(filename, batch_size):
    dataset = WordsDataset(filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    return dataloader
