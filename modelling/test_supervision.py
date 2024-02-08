import torch
import torch.nn as nn

if __name__ == "__main__":
    x = torch.rand(4, 100)
    y = torch.rand(4, 1024,1024)
    transform_a = nn.Linear(100,2)
    transform_b = nn.Linear(1024,2)
    r1 = transform_a(x)
    r2 = transform_b(y)
    print(r1.shape)
    print(r2.shape)
    r1 = r1.unsqueeze(1)
    print(r1.shape)
    
    z = r1 + r2
    
    # do softmax over the last dimension
    print(z.shape)
    z = nn.functional.softmax(z, dim=-1)
    features = torch.rand(4,1024,1024)
    #do pointwise multiplication
    print(z.shape)
    print(features.shape)
    gate_1 = z[:,:,:1]
    gate_2 = z[:,:,1:]
    print(gate_1.shape)
    print(gate_2.shape)
    #gate_1 = gate_1.unsqueeze(-1)
    #gate_2 = gate_2.unsqueeze(-1)
    print(gate_1.shape)
    print(gate_2.shape)

    z1 = gate_1 * features
    z2 = gate_2 * features
    print(z1.shape)
    print(z2.shape)

    a = torch.rand(4,2,2,requires_grad=True)
    b = torch.rand(4,2,1,requires_grad=True)
    c = a * b
    print(c.shape)
    print(a.shape)
    print(b.shape)
    print(a)
    print(b)
    print(c)

    