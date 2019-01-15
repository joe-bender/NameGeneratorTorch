import torch
import torch.nn as nn
import torch.optim as optim
import data
import helpers
from NamesRNN import NamesRNN
from hyperparameters import hps
import random

model = NamesRNN()
names = data.get_names(hps['filename'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hps['learning_rate'])

batch_size = hps['batch_size']

#for t in range(hps['epochs']):
while True:
    batch = random.sample(names, batch_size)
    batch_losses = []
    for name in batch:
        xs, ys = helpers.name_to_xy(name)

        seq_losses = []
        # initialize the hidden output
        hidden = None
        for i_seq in range(len(name)):
            x, y = xs[i_seq], ys[i_seq]
            y_pred, hidden = model(x, hidden)
            # reshape prediction for compatibility with CrossEntropyLoss
            y_pred = y_pred.view(1, -1)
            this_seq_loss = criterion(y_pred, y)
            seq_losses.append(this_seq_loss)
        seq_loss = torch.mean(torch.stack(seq_losses))
        batch_losses.append(seq_loss)
    batch_loss = torch.mean(torch.stack(batch_losses))

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    # print loss at intervals
    if (t+1) % hps['print_every'] == 0 or t == 0:
        print(t+1, batch_loss.item())

    # save model at intervals
    if (t+1) % hps['save_every'] == 0:
        torch.save(model.state_dict(), 'models/model.pt')
