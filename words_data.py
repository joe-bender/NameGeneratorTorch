import torch
from torch.utils.data import Dataset, DataLoader
from get_data import get_data

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
