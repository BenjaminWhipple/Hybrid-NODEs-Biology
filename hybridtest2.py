'''
Notes to self:

This took about 3 minutes to run on balanced (141 seconds). Not terrible by any means.

Possible extensions: Implement other systems, including maybe a more complex Epidemiological model.

TODO:

Implement NODE case.

Implement extension of data so that we can have different training/testing.

Implement more complex functional forms for interaction.
Implement more complex systems (more compartments)

Use RMSE as output.
Vary hyperparameters, (iterations, data size, batch time, batch size, network sizes)

#Paper idea, NN defined control of population models.

Create function to construct a network of given size.

Collect data, RMSE, training time, etc.

Vary model parameters across space which leads to desireable cyclical dynamics

#Find papers which benchmark architectures

'''

### Boilerplate Code

# Import packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import torchdiffeq
import time
# Install torchdiffeq from git
#!pip install git+https://github.com/rtqichen/torchdiffeq.git


# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# If GPU acceleration is available
gpu=0
global device
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Set up training parameters

# Training parameters
#niters=2000        # training iterations
niters=1000
data_size=1000     # samples in dataset
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch

### DATA GENERATION ###

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


#NOTE: Just pull from Rackaukus' paper.
# Initial condition, time span & parameters
true_y0 = torch.tensor([[5.0,5.0]]).to(device)
t = torch.linspace(0., 5., data_size).to(device)
p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)


# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(LV(), true_y0, t, method='dopri5')
print("Data generated.")

# Add noise (mean = 0, std = 0.1)
true_y *= (1 + torch.randn(data_size,1,2)/20.)

#NOTE: This function is worth getting a better understanding of.
# Batch function
def get_batch(batch_time, batch_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

### MODELS ###

# Data-driven model
class neuralODE(nn.Module):

    def __init__(self):
        super(neuralODE, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.5)

    def forward(self, t, y):

        return self.net(y)

# Integrated first-principles/data-driven model
class hybridODE(nn.Module):

    def __init__(self, p0):
        super(hybridODE, self).__init__()

        self.paramsODE = [p0[0],p0[-1]]#nn.Parameter(p0)
        self.alpha = self.paramsODE[0]     #mM min-1
        self.delta = self.paramsODE[1]

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,2)[:,0]
        S2 = y.view(-1,2)[:,1]

        dS1 = self.alpha*S1 #for dimensions
        dS2 = -self.delta*S2
        return (torch.stack([dS1, dS2], dim=1).view(-1,1,2) + self.net(y)).to(device)


### TRAINING ###

# Initialization
lr = 1e-2                                       # learning rate
#p0 = torch.tensor([2., 120., 5., 14., 90., 1., 15., 2., 15., 4., 0.4, 0.2, 2., 3.]).to(device)
model = hybridODE(p).to(device)                # choose type of model to train (pureODE(p0), neuralODE(), hybridODE(p0))
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler

start = time.time()

print("Starting training.")
for it in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size)
    pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (it) % 100 == 0:
        print('Iteration: ', it, '/', niters)

end = time.time()

### VISUALIZATION ###

pred_y = odeint(model, true_y0.view(1,1,2), t, method='rk4').view(-1,1,2)

plt.figure(figsize=(20, 10))
plt.plot(t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy())
plt.plot(t.detach().cpu().numpy(), true_y[:,0].cpu().numpy(), 'o',alpha=0.1)
plt.show()

print(f'Time taken = {end-start} seconds')