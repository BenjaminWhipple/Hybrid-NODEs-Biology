### Boilerplate Code

data_size = 100 # Number of points to train on.

# Import packages
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
class LinkedSIR(nn.Module):
    
    def __init__(self):
        super(LinkedSIR, self).__init__()
        # beta1, beta2 : transmission rates for sites 1 and 2 resp.
        # gamma1, gamma2 : recovery rates for sites 1 and 2 resp.
        # mu12, mu21 : migration rates between sites 1 to 2 and sites 2 to 1 respectively
        self.beta1 = 0.3
        self.beta2 = 0.2
        self.gamma1 = 0.1
        self.gamma2 = 0.1
        self.mu12 = 0.01
        self.mu21 = 0.01

    def forward(self, t, y):
        S1 = y.view(-1,6)[:,0]
        I1 = y.view(-1,6)[:,1]
        R1 = y.view(-1,6)[:,2]
        S2 = y.view(-1,6)[:,3]
        I2 = y.view(-1,6)[:,4]
        R2 = y.view(-1,6)[:,5]
        
        N1 = S1 + I1 + R1  # Total for site 1 (not necessarily needed but useful for clarity)
        N2 = S2 + I2 + R2  # Total for site 2
        
        dS1 = -self.beta1 * S1 * I1 / N1 - self.mu12 * S1 + self.mu21 * S2
        dI1 = self.beta1 * S1 * I1 / N1 - self.gamma1 * I1 - self.mu12 * I1 + self.mu21 * I2
        dR1 = self.gamma1 * I1 - self.mu12 * R1 + self.mu21 * R2
        dS2 = -self.beta2 * S2 * I2 / N2 - self.mu21 * S2 + self.mu12 * S1
        dI2 = self.beta2 * S2 * I2 / N2 - self.gamma2 * I2 - self.mu21 * I2 + self.mu12 * I1
        dR2 = self.gamma2 * I2 - self.mu21 * R2 + self.mu12 * R1 

        return torch.stack([dS1, dI1, dR1, dS2, dI2, dR2], dim=1).to(device)


### DATA GENERATION
train_y0 = torch.tensor([[0.49, 0.01, 0.0, 0.50, 0.0, 0.0]]).to(device)
test_y0 = torch.tensor([[0.4,0.1, 0.0, 0.45, 0.05, 0.0]]).to(device)

t = torch.linspace(0., 100., data_size+1).to(device)

# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    train_y = odeint(LinkedSIR(), train_y0, t, method='dopri5')
    test_y = odeint(LinkedSIR(), test_y0, t, method='dopri5')
print("Data generated.")


# Add noise
train_y *= (1 + torch.randn(data_size+1,1,6)/20.)
test_y *= (1 + torch.randn(data_size+1,1,6)/20.)

torch.save(train_y,'LinkedSIR_train_data.pt')
torch.save(test_y,'LinkedSIR_test_data.pt')

