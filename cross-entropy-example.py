import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
input = torch.randn(8, 5)
print(input)
target = torch.LongTensor(8).random_(5)
print(target)
print(target.shape)
output = loss(input, target)
print(output)
