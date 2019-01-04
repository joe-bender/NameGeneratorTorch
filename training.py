import torch
import torch.nn as nn
import torch.optim as optim
from data_funcs import get_dataloader
from NamesRNN import NamesRNN
from hyperparameters import hps

model = NamesRNN()
dataloader = get_dataloader('names2017.csv', hps['batch_size'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hps['learning_rate'])

for t in range(hps['epochs']):
    loss = None
    for x, y in dataloader:
        y_pred, _ = model(x)

        # take mean of losses over each step of the sequence
        losses = []
        seq_length = y_pred.shape[0]
        for i in range(seq_length):
            loss = criterion(y_pred[i], y[i])
            losses.append(loss)
        losses = torch.stack(losses)
        loss = torch.mean(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print loss at intervals
    if (t+1) % hps['print_every'] == 0 or t == 0:
        print(t+1, loss.item())

torch.save(model.state_dict(), 'models/model.pt')
