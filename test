import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


""" testing how you can detach one head of a model from loss """
torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = nn.Linear(3,2)
        self.head1 = nn.Linear(2,5)
        self.head2 = nn.Linear(2,5)
        
    
    def forward(self, x):
        x = self.backbone(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        return 0.5 * x1 + 0.5 * x2, x1, x2

def cross_entropy_loss(pred, target):
    loss = nn.CrossEntropyLoss()

    return loss(pred, target)

def distance_loss(pred, target):
    return nn.MSELoss()(pred, target)

input = torch.randn(2,3)
labels = torch.empty(2, dtype=torch.long).random_(5)
print(labels)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
x, x1, x2 = model(input)
x3 = x1.detach()
print(x.shape)
print(labels.shape)
loss1 = cross_entropy_loss(x, labels)
loss2 = distance_loss(x3, x2)
new_loss = loss1 + loss2
#new_loss = loss1
new_loss.backward()
optimizer.step()
print(model.head1)
print(model.head1.weight.grad)
print(model.head2)
print(model.head2.weight.grad)
print(model.backbone)
print(model.backbone.weight.grad)





