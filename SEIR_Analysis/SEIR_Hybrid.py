### Boilerplate Code

# Import packages
import pandas as pd
#import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#import random
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
global device
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Utility Fns

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

### End Boilerplate code.

### DATA GENERATION ###

### TRUE MODEL
class SEIR(nn.Module):
    
    def __init__(self):
        super(SEIR, self).__init__()
        # beta1, beta2 : transmission rates for sites 1 and 2 resp.
        # gamma1, gamma2 : recovery rates for sites 1 and 2 resp.
        # mu12, mu21 : migration rates between sites 1 to 2 and sites 2 to 1 respectively
        self.beta = 0.3
        self.sigma = 0.2
        self.gamma = 0.1

    def forward(self, t, y):
        S = y.view(-1,4)[:,0]
        E = y.view(-1,4)[:,1]
        I = y.view(-1,4)[:,2]
        R = y.view(-1,4)[:,3]
        
        N = S+E+I+R
        
        dS = -self.beta * S * I / N
        dE = self.beta*S*I/N - self.sigma*E
        dI = self.sigma*E - self.gamma*I
        dR = self.gamma * I
        return torch.stack([dS, dE, dI, dR], dim=1).to(device)

### PURE NODE
class neuralODE(nn.Module):

    def __init__(self,structure):
        super(neuralODE, self).__init__()

        self.net = self.make_nn(structure)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.5)

    def forward(self, t, y):

        return self.net(y)
    
    def make_nn(self, structure):
        '''
        Structure should contain:
        1. Input size
        2. Output size 
        3. Size for each hidden layers list of (10,20,30,40,50) of length equal to num of hidden layers
        Maybe? 3. Activation function for each layer (Tanh)
        '''
        input_dim = structure[0]
        output_dim = structure[1]
        num_layers = len(structure[2])
        hidden_sizes = structure[2]
        modules = []
        
        for i in range(num_layers):
            if i==0:
                modules.append(nn.Linear(input_dim,hidden_sizes[i]))
                modules.append(nn.Tanh())
            
            elif i<num_layers:
                modules.append(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
                modules.append(nn.Tanh())
            
            else:
                pass

        modules.append(nn.Linear(hidden_sizes[-1],output_dim))
        
        return nn.Sequential(*modules)

class known_params_hybridODE(nn.Module):

    def __init__(self,structure):
        super(known_params_hybridODE, self).__init__()
        
        self.beta = 0.3
        self.sigma = 0.2
        self.gamma = 0.1
        

        self.net = self.make_nn(structure)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S = y.view(-1,4)[:,0]
        E = y.view(-1,4)[:,1]
        I = y.view(-1,4)[:,2]
        R = y.view(-1,4)[:,3]
        
        N = S+E+I+R
        
        dS = self.beta*S*I
        dE = self.beta*S*I
        dI = - self.gamma*I
        dR = self.gamma * I
        
        return (torch.stack([dS, dE, dI, dR], dim=1).view(-1,1,4) + self.net(y)).to(device)
        
        #return (torch.stack([dS1, dS2], dim=1).view(-1,1,2) + self.net(y)).to(device)
    
    def make_nn(self, structure):
        '''
        Structure should contain:
        1. Input size
        2. Output size 
        3. Size for each hidden layers list of (10,20,30,40,50) of length equal to num of hidden layers
        Maybe? 3. Activation function for each layer (Tanh)
        '''
        input_dim = structure[0]
        output_dim = structure[1]
        num_layers = len(structure[2])
        hidden_sizes = structure[2]
        modules = []
        

        #print(hidden_sizes)
        for i in range(num_layers):
            print(hidden_sizes[i])
            if i==0:
                #Add input layer
                #print(i)
                modules.append(nn.Linear(input_dim,hidden_sizes[i]))
                modules.append(nn.Tanh())
            
            elif i<num_layers:
                #print(i)
                
                #print(hidden_sizes[i-1])
                #print(hidden_sizes[i])
                modules.append(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
                modules.append(nn.Tanh())
            
            else:
                pass
        #print(i)
        modules.append(nn.Linear(hidden_sizes[-1],output_dim))
        
        #print(modules)
        return nn.Sequential(*modules)

### DATA GENERATION

### Set up training parameters

# Training parameters
niters_initial = 5000
niters=1000        # training iterations
#niters=1000
data_size= 100 # samples in dataset
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch

reg_param = 0.0

# Initial condition, time span & parameters
#true_y0 = torch.tensor([[5.0,5.0]]).to(device)
true_y0 = torch.tensor([[0.99, 0.01, 0.0, 0.0]]).to(device)
t = torch.linspace(0., 100., data_size+1).to(device)
#p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(SEIR(), true_y0, t, method='dopri5')
print("Data generated.")


# Add noise (mean = 0, std = 0.1)
true_y *= (1 + torch.randn(data_size+1,1,4)/20.)





# Batch function for training
def get_batch(batch_time, batch_size,data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y


model = known_params_hybridODE((4,4,[10,10])).to(device)

start = time.time()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) #optional learning 
for it in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
    pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
    
    #Now we are going to try to incorporate a better loss fn.
    loss = torch.mean(torch.abs(pred_y - batch_y))
    #MAE + L1 regularization of NN params.
    #loss = torch.mean(torch.abs(pred_y - batch_y)) + torch.sum(NN Params.)
    #Need to figure out NN params.
    MAE = torch.mean(torch.abs(pred_y - batch_y))
    L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())]))
    loss = MAE + L1_Reg

    loss.backward()
    optimizer.step()
    scheduler.step()

    #'''

    if (it) % 10 == 0:
        print('Iteration: ', it, '/', niters)
        print(loss)
    
    #'''


end = time.time()

TimeTaken = end-start
print(f'Time Taken: {TimeTaken}')

NumParams = get_n_params(model)

pred_y = odeint(model, true_y0.view(1,1,4), t, method='rk4').view(-1,1,4)

SSE = float(torch.sum(torch.square(pred_y - true_y)))
RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

plt.figure(figsize=(20, 10))
plt.plot(t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy())
plt.plot(t.detach().cpu().numpy(), true_y[:,0].cpu().numpy(), 'o',alpha=0.3)
plt.show()

#print(pred_y)

