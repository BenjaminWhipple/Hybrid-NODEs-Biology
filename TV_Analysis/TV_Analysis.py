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
import pandas as pd
import itertools
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
class TV(nn.Module):

    def __init__(self,params):
        super(LV, self).__init__()
        #1.3, 0.9, 0.8, 1.8
        self.p = params[0]
        self.kV = params[1]
        self.cV = params[2]
        self.r = params[3]
        self.kE = params[4]
        self.cE = params[5]
        self.V0 = params[6]
        self.E0 = params[7]
        self.sE = self.cE*self.E0

    def forward(self, t, y):
        V = y.view(-1,2)[:,0]
        E = y.view(-1,2)[:,1]

        dV = self.p*V*(1-V/self.kV) - self.cV*V*E
        dE = self.r*E*(V/(V+self.kE)) - self.cE*E + self.sE
        return torch.stack([dV, dE], dim=1).to(device)

### PURE NODE
class neuralODE(nn.Module):

    def __init__(self,structure=(2,2,(50,20))):
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
        

        #print(hidden_sizes)
        for i in range(num_layers):
            #print(hidden_sizes[i])
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


# Integrated first-principles/data-driven model
class hybridODE(nn.Module):

    def __init__(self, p0,structure):
        super(hybridODE, self).__init__()

        self.paramsODE = [p0[0],p0[-1]]#nn.Parameter(p0)
        self.alpha = self.paramsODE[0]     #mM min-1
        self.delta = self.paramsODE[1]

        self.net = self.make_nn(structure)

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
#niters=2000        # training iterations
niters=1000
data_size=1000     # samples in dataset
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch

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

#TODO: Normalize the model data, see if this improves the fitting.
#

# Batch function for training
def get_batch(batch_time, batch_size,data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

### TRAINING ###
'''
hidden layers: 1,2 (N)
layer sizes in (20,35,50)^N
niters = (750,1000,1250)
data_size = (100, 550, 1000)
batch_time = (8, 16, 32)
batch_size = (64, 128, 256)
'''
def LV_neuralODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size):
    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In Neural ODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    t = torch.linspace(0., 5., data_size).to(device)
    p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(LV(), true_y0, t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    true_y *= (1 + torch.randn(data_size,1,2)/20.)

    model = neuralODE((2,2,layer_sizes)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
        pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()
        scheduler.step()

        '''

        if (it) % 100 == 0:
            print('Iteration: ', it, '/', niters)
        
        '''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    NumParams = get_n_params(model)

    pred_y = odeint(model, true_y0.view(1,1,2), t, method='rk4').view(-1,1,2)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    return [SSE, RMSE, TimeTaken, NumParams]

def LV_hybridODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size):

    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In HybridODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    t = torch.linspace(0., 5., data_size).to(device)
    p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(LV(), true_y0, t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    true_y *= (1 + torch.randn(data_size,1,2)/20.)

    model = hybridODE(p,(2,2,layer_sizes)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
        pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()
        scheduler.step()

        '''

        if (it) % 100 == 0:
            print('Iteration: ', it, '/', niters)
        
        '''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    NumParams = get_n_params(model)

    pred_y = odeint(model, true_y0.view(1,1,2), t, method='rk4').view(-1,1,2)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    return [SSE, RMSE, TimeTaken, NumParams]

print(LV_neuralODE_experiment((20,20),1000,1000,16,256))
#print(f'Time taken = {end-start} seconds')

'''
I probably need to write an experimentation loop over the rest of the code.

Output: RMSE, Time Spent, Num Params, layer sizes, num hidden layers, niters, data_size, batch_time, batch_size.

Input: hidden layers layer sizes, niters, data_size, batch_time, batch_size.

For same given hyperparameter: run 10 replicates?

hidden layers: 1,2 (N)
layer sizes in (20,35,50)^N
niters = (750,1000,1250)
data_size = (100, 550, 1000)
batch_time = (8, 16, 32)
batch_size = (64, 128, 256)

Works out to be 1458*replicates number of evaluations
-> 5 replicates * 1458 combinations * avg 1 minute per run:
-> 5 days.

How do we want to break into multiple files?

Maybe run experiments for each data_size, model, and niters. After which, save into csv.

Shifts into chunks of 9*5*3 runs ~ 4 hrs.
filenames can be "<model type>_<data_size>_<niters>.csv
Columns would be:
1. REPLICATE (1-5)
2. SSE
3. RMSE
4. Time taken
5. num params
6. num hidden layers
7. layer dimensions
8. niters
9. data size
10. batch time
11. batch size

'''
hidden_layers = [1,2]
layer_size = [20,35,50]
layer_sizes = itertools.product(layer_size,layer_size)
niters = [750,1000,1250]
data_size = [100, 550, 1000]
batch_time = [8, 16, 32]
batch_size = [64, 128, 256]

results = []
#dims = (20,20)
counter = 0
for dims in layer_sizes:
    print(dims)
    for i in data_size:
        for j in niters:
            for rep in range(5):
                counter += 1
                print(f'Simulation {counter}')
                #result = LV_hybridODE_experiment(dims,j,i,16,256)
                result = LV_neuralODE_experiment(dims,j,i,16,256)
                results.append([rep,*result, 2, str(dims).replace(',',''),j,i,16,256])

results_df = pd.DataFrame(results, columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','Hidden Layers','Layer Dimensions','niters','data size','batch time','batch size'])
results_df.to_csv('Results_neuralODE.csv',index=False)
