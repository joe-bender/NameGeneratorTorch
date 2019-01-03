import torch
import torch.nn as nn
import torch.optim as optim
from data_funcs import get_dataloader

# hyper-parameters
D_in = 26
D_out = 26
layers = 2
lr = .01
seq_length = 5
batch_size = 64
C = D_out # number of categories for classifiction
epochs = 200
print_every = 1

model = nn.LSTM(D_in, D_out, layers)
dataloader = get_dataloader('names2017.csv', batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for t in range(epochs):
    loss = 0
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
    # print loss occasionally
    if (t+1) % print_every == 0 or t == 0:
        print(t+1, loss.item())

torch.save(model.state_dict(), 'models/model.pt')
