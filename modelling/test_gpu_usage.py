import torch
import os

a = torch.rand(10,20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
a = a.to(device)
while True:
    pass