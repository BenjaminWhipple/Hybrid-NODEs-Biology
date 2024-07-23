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

class Seasonal_LV(nn.Module):
    def __init__(self):
        super(Seasonal_LV, self).__init__()
        self.alpha = 0.5
        self.beta = 1.5
        self.gamma = 3.0
        self.delta = 1.0
        self.L = 1.0
        
    def forward(self, t, y):
        S1 = y.view(-1,4)[:,0]
        S2 = y.view(-1,4)[:,1]
        S3 = y.view(-1,4)[:,2]
        T = y.view(-1,4)[:,3] # We use this as a proxy for time due to difficulties with implementing time dependency in the torchdiffeq interface.
        
        #dS1 = self.alpha*(1+torch.sin(T/self.L))*S1 - self.beta*S1*S2
        dS1 = self.alpha*S1 - self.beta*S1*S2
        dS2 = self.beta*S1*S2 - self.gamma*S2*S3
        dS3 = self.gamma*S2*S3 - self.delta*S3
        dT = torch.tensor([1.0])
        return torch.stack([dS1, dS2, dS3, dT], dim=1).to(device)

true_y0 = torch.tensor([[1.0,1.0,1.0,0.0]]).to(device)
t = torch.linspace(0., 9., 91).to(device)
p = torch.tensor([0.5,1.5,3.0,1.0,1.0]).to(device)

print("Generating data.")
with torch.no_grad():
    gen_y = odeint(Seasonal_LV(), true_y0, t, method='dopri5')
print("Data generated.")

## NOTE: The 4th column of this data is redundant and can be discarded.

# Add noise
y = gen_y * (1 + torch.randn(91,1,4)/20.)

torch.save(y,'Seasonal_LV_data.pt')

print(y[:,0,0])

plt.figure(figsize=(8,6))
plt.plot(t, y[:,0,0].to('cpu'),'o')
plt.plot(t, y[:,0,1].to('cpu'),'o')
plt.plot(t, y[:,0,2].to('cpu'),'o')
plt.savefig("noisy_data.png")

plt.figure(figsize=(8,6))
plt.plot(t, y[:,0,1].to('cpu'),'o')
plt.plot(t, y[:,0,2].to('cpu'),'o')
plt.savefig("available_data.png")

plt.figure(figsize=(8,6))
plt.plot(t, gen_y[:,0,0].to('cpu'))
plt.plot(t, gen_y[:,0,1].to('cpu'))
plt.plot(t, gen_y[:,0,2].to('cpu'))
plt.savefig("data.png")