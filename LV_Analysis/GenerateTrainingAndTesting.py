data_size=50     # samples in dataset

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchdiffeq
import time
import sys

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# If GPU acceleration is available
gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### TRUE MODEL
class LV(nn.Module):

    def __init__(self):
        super(LV, self).__init__()
        #1.3, 0.9, 0.8, 1.8
        self.alpha = 1.3
        self.beta = 0.9
        self.gamma = 0.8
        self.delta = 1.8

    def forward(self, t, y):
        S1 = y.view(-1,2)[:,0]
        S2 = y.view(-1,2)[:,1]

        dS1 = self.alpha*S1 - self.beta*S1*S2
        dS2 = self.gamma*S1*S2 - self.delta*S2
        return torch.stack([dS1, dS2], dim=1).to(device)

true_y0 = torch.tensor([[5.0,5.0]]).to(device)
t = torch.linspace(0., 15., int((3/2)*data_size)+1).to(device)
p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(LV(), true_y0, t, method='dopri5')
print("Data generated.")

# Add noise (mean = 0, std = 0.1)
true_y *= (1 + torch.randn(int((3/2)*data_size)+1,1,2)/20.)

# Save data
torch.save(true_y,'LV_data.pt')
