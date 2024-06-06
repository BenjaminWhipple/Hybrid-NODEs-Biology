data_size=150

# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random
import torchdiffeq
import time

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class glycolysis(nn.Module):
    def __init__(self):
        super(glycolysis, self).__init__()
        self.J0 = 2.5    #mM min-1
        self.k1 = 100.   #mM-1 min-1
        self.k2 = 6.     #mM min-1
        self.k3 = 16.    #mM min-1
        self.k4 = 100.   #mM min-1
        self.k5 = 1.28   #mM min-1
        self.k6 = 12.    #mM min-1
        self.k = 1.8     #min-1
        self.kappa = 13. #min-1
        self.q = 4.
        self.K1 = 0.52   #mM
        self.psi = 0.1
        self.N = 1.      #mM
        self.A = 4.      #mM

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0 - (self.k1*S1*S6)/(1 + (S6/self.K1)**self.q)
        dS2 = 2. * (self.k1*S1*S6)/(1 + (S6/self.K1)**self.q) - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = -2. * (self.k1 * S1 * S6) / (1 + (S6 / self.K1)**self.q) + 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).to(device)


true_y0 = torch.tensor([[1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1]]).to(device)
t = torch.linspace(0., 6., int((3/2)*data_size)+1).to(device)
p = torch.tensor([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.]).to(device)

print("Generating data.")
with torch.no_grad():
    true_y = odeint(glycolysis(), true_y0, t, method='dopri5')
print("Data generated.")


# Add noise
true_y *= (1 + torch.randn(int((3/2)*data_size)+1,1,7)/20.)

torch.save(true_y,'Glycolysis_data.pt')

