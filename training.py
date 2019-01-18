"""Train the RNN model on random batches of names from the given text file

Hyperparameters are taken from hyperparameters.py
The model takes input in the form of (sequence_index, batch_index, onehot_tensor),
but we give it tensors of size (1, 1, 27) because it only gets one letter at a
time. A certain number of names, given by batch_size, is sampled from the full
list of names. Then the letters from each name from the minibatch are sent through
the network one at a time, with the hidden output from one letter being the hidden
input for the next letter. The losses from each name in the minibatch are averaged
and the final loss for each minibatch is calculated from this average.
CrossEntropyLoss is used because this is a categorization task, since we're
predicting the next letter of the name given previous letters.
"""

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

for t in range(hps['epochs']):
    batch = random.sample(names, batch_size)
    # keep losses from each batch to be averaged later
    batch_losses = []
    for name in batch:
        xs, ys = helpers.name_to_xy(name)

        # keep losses from each sequence (name) to be averaged later
        seq_losses = []
        # start with hidden state of zeros for each new sequence (name)
        hidden = None
        # loop through each letter (sequence index) of the name
        for i_seq in range(len(name)):
            x, y = xs[i_seq], ys[i_seq]
            y_pred, hidden = model(x, hidden)
            # reshape prediction for compatibility with CrossEntropyLoss
            y_pred = y_pred.view(1, -1)
            this_seq_loss = criterion(y_pred, y)
            seq_losses.append(this_seq_loss)
        # get the mean of all losses from the sequence
        seq_loss = torch.mean(torch.stack(seq_losses))
        batch_losses.append(seq_loss)
    # get the mean of all losses from the batch of names
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
