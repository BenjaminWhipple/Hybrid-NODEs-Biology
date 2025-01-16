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

#import pysr
#pysr.install()
# Install torchdiffeq from git
#!pip install git+https://github.com/rtqichen/torchdiffeq.git


#Expect input of python file iteration_num
iteration_num = sys.argv[1]

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

#TODO: Implement pure ODE as pytorch module.

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

        #We initialize the parameters this way in order to actually train them
        self.paramsODE = nn.Parameter(torch.tensor([p0[0],p0[-1]]))

        self.alpha = self.paramsODE[0]     #mM min-1
        self.delta = self.paramsODE[-1]

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

class known_params_hybridODE(nn.Module):

    def __init__(self, p0,structure):
        super(known_params_hybridODE, self).__init__()

        #We initialize the parameters this way in order to actually train them
        #self.paramsODE = nn.Parameter(torch.tensor([p0[0],p0[-1]]))

        self.alpha = p0[0]     #mM min-1
        self.delta = p0[-1]

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

class LV_pureODE(nn.Module):

    def __init__(self, p0):
        super(LV_pureODE, self).__init__()

        #We initialize the parameters this way in order to actually train them
        self.paramsODE = nn.Parameter(torch.tensor(p0))

        self.alpha = self.paramsODE[0]
        self.beta = self.paramsODE[1]
        self.gamma = self.paramsODE[2]
        self.delta = self.paramsODE[3]

    def forward(self, t, y):
        S1 = y.view(-1,2)[:,0]
        S2 = y.view(-1,2)[:,1]

        dS1 = self.alpha*S1 - self.beta*S1*S2
        dS2 = self.gamma*S1*S2 - self.delta*S2
        return (torch.stack([dS1, dS2], dim=1).view(-1,1,2)).to(device)
    
### DATA GENERATION

### Set up training parameters

# Training parameters
niters=5000        # training iterations
#niters=1000
data_size=50     # samples in dataset
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch

# Initial condition, time span & parameters
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
def LV_puremech_experiment(niters,data_size,batch_time,batch_size):
    print(f'Data size: {data_size}')

    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In pure mech Experiment: {[niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    t = torch.linspace(0., 10., data_size).to(device)
    true_t = torch.linspace(0.,15.,int((3/2)*data_size+1)).to(device)
    train_t = torch.linspace(0.,10.,(data_size)+1).to(device)
    p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(LV(), true_y0, true_t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    #measured_y = (1 + torch.randn(int((3/2)*data_size+1),1,2)/20.)
    #train_y = measured_y[:data_size+1]
    #print(train_y)
    #print(len(train_y))

    #model = hybridODE(p,(2,2,layer_sizes)).to(device)
    model = LV_pureODE([1.0,1.0,1.0,1.0]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
        pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #Need a better loss function
        #MAE + L1 regularization of NN params.
        MAE = torch.mean(torch.abs(pred_y - batch_y))
        #L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())[1:]]))
        #loss = MAE + L1_Reg
        loss = MAE

        loss.backward()
        optimizer.step()
        scheduler.step()

        #print(model.paramsODE[0])

        #'''

        if (it) % 500 == 0:
            print('Iteration: ', it, '/', niters)
        
        #'''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    NumParams = get_n_params(model)

    pred_y = odeint(model, true_y0.view(1,1,2), true_t, method='rk4').view(-1,1,2)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    return [SSE, RMSE, TimeTaken, NumParams, pred_y]

def LV_neuralODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size,reg_param=1.0):
    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In Neural ODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    t = torch.linspace(0., 10., data_size).to(device)
    true_t = torch.linspace(0.,15.,int((3/2)*data_size+1)).to(device)
    train_t = torch.linspace(0.,10.,(data_size)+1).to(device)
    p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(LV(), true_y0, true_t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    #measured_y = (1 + torch.randn(int((3/2)*data_size)+1,1,2)/20.)
    #train_y = measured_y[:data_size+1]
    #print(train_y)
    #print(len(train_y))

    model = neuralODE((2,2,layer_sizes)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
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

        if (it) % 500 == 0:
            print('Iteration: ', it, '/', niters)
        
        #'''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    NumParams = get_n_params(model)

    pred_y = odeint(model, true_y0.view(1,1,2), true_t, method='rk4').view(-1,1,2)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    #We should also conduct a PySR regression on the NN.
    """
    #Compute the RMSE between the recovered mechanism and the actual mechanism
    with torch.no_grad():
        #Want to get the response over the range
        max1 = max(pred_y[:,0,0])
        max2 = max(pred_y[:,0,1])
        min1 = min(pred_y[:,0,0])
        min2 = min(pred_y[:,0,1])

        STATE_SAMPLES = 100

        #The +0 converts the tensor to a float.
        state1 = torch.linspace(min1+0,max1+0,STATE_SAMPLES)
        state2 = torch.linspace(min2+0,max2+0,STATE_SAMPLES)

        test = torch.stack([state1,state2],dim=0)
        true_model = lambda x: torch.tensor([-0.9*x[0]*x[1],0.8*x[0]*x[1]])
        
        pred_response = torch.stack([model.net(test[:,i]) for i in range(test.shape[1])])
        true_response = torch.stack([true_model(test[:,i]) for i in range(test.shape[1])])

        #Use a fairly focussed regressor.
        model=pysr.PySRRegressor(
            niterations=40,
            binary_operators=["+","*"],
            loss="loss(prediction,target)=(prediction-target)^2"
            )
        
        test_transpose =torch.transpose(test,0,1)

        model.fit(test_transpose,pred_response)

        err = torch.mean(torch.sqrt(torch.square(torch.tensor(model.predict(test_transpose))-true_response)))
    """
    #return [SSE, RMSE, TimeTaken, NumParams, pred_y,err]
    return [SSE, RMSE, TimeTaken, NumParams, pred_y]


def LV_hybridODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size,reg_param=1.0):
    print(f'Data size: {data_size}')

    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In HybridODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    t = torch.linspace(0., 10., data_size).to(device)
    true_t = torch.linspace(0.,15.,int((3/2)*data_size+1)).to(device)
    train_t = torch.linspace(0.,10.,(data_size)+1).to(device)
    p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(LV(), true_y0, true_t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    #measured_y = (1 + torch.randn(int((3/2)*data_size+1),1,2)/20.)
    #train_y = measured_y[:data_size+1]
    #print(train_y)
    #print(len(train_y))

    #model = hybridODE(p,(2,2,layer_sizes)).to(device)
    model = hybridODE([1.3,1.0,1.0,1.8],(2,2,layer_sizes)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
        pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #Need a better loss function
        #MAE + L1 regularization of NN params.
        MAE = torch.mean(torch.abs(pred_y - batch_y))
        L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())[1:]]))
        loss = MAE + L1_Reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        #print(model.paramsODE[0])

        #'''

        if (it) % 500 == 0:
            print('Iteration: ', it, '/', niters)
        
        #'''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    NumParams = get_n_params(model)

    pred_y = odeint(model, true_y0.view(1,1,2), true_t, method='rk4').view(-1,1,2)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    #We should also conduct a PySR regression on the NN.
    
    #Compute the RMSE between the recovered mechanism and the actual mechanism
    """
    with torch.no_grad():
        #Want to get the response over the range
        max1 = max(pred_y[:,0,0])
        max2 = max(pred_y[:,0,1])
        min1 = min(pred_y[:,0,0])
        min2 = min(pred_y[:,0,1])

        STATE_SAMPLES = 100

        #The +0 converts the tensor to a float.
        state1 = torch.linspace(min1+0,max1+0,STATE_SAMPLES)
        state2 = torch.linspace(min2+0,max2+0,STATE_SAMPLES)

        test = torch.stack([state1,state2],dim=0)
        true_model = lambda x: torch.tensor([-0.9*x[0]*x[1],0.8*x[0]*x[1]])
        
        pred_response = torch.stack([model.net(test[:,i]) for i in range(test.shape[1])])
        true_response = torch.stack([true_model(test[:,i]) for i in range(test.shape[1])])

        #Use a fairly focussed regressor.
        model=pysr.PySRRegressor(
            niterations=40,
            binary_operators=["+","*"],
            loss="loss(prediction,target)=(prediction-target)^2"
            )
        
        test_transpose =torch.transpose(test,0,1)

        model.fit(test_transpose,pred_response)

        err = torch.mean(torch.sqrt(torch.square(torch.tensor(model.predict(test_transpose))-true_response)))
    """

    #return [SSE, RMSE, TimeTaken, NumParams, pred_y,err]
    return [SSE, RMSE, TimeTaken, NumParams, pred_y]

def LV_known_params_hybridODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size,reg_param=1.0):
    print(f'Data size: {data_size}')

    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In Known Params HybridODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    t = torch.linspace(0., 10., data_size).to(device)
    true_t = torch.linspace(0.,15.,int((3/2)*data_size+1)).to(device)
    train_t = torch.linspace(0.,10.,(data_size)+1).to(device)
    p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(LV(), true_y0, true_t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    #measured_y = (1 + torch.randn(int((3/2)*data_size+1),1,2)/20.)
    #train_y = measured_y[:data_size+1]
    #print(train_y)
    #print(len(train_y))

    #model = hybridODE(p,(2,2,layer_sizes)).to(device)
    model = known_params_hybridODE([1.3,1.0,1.0,1.8],(2,2,layer_sizes)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
        pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #Need a better loss function
        #MAE + L1 regularization of NN params.
        MAE = torch.mean(torch.abs(pred_y - batch_y))
        L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())[1:]]))
        loss = MAE + L1_Reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        #print(model.paramsODE[0])

        #'''

        if (it) % 500 == 0:
            print('Iteration: ', it, '/', niters)
        
        #'''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    NumParams = get_n_params(model)

    pred_y = odeint(model, true_y0.view(1,1,2), true_t, method='rk4').view(-1,1,2)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    #We should also conduct a PySR regression on the NN.
    
    #Compute the RMSE between the recovered mechanism and the actual mechanism
    """
    with torch.no_grad():
        #Want to get the response over the range
        max1 = max(pred_y[:,0,0])
        max2 = max(pred_y[:,0,1])
        min1 = min(pred_y[:,0,0])
        min2 = min(pred_y[:,0,1])

        STATE_SAMPLES = 100

        #The +0 converts the tensor to a float.
        state1 = torch.linspace(min1+0,max1+0,STATE_SAMPLES)
        state2 = torch.linspace(min2+0,max2+0,STATE_SAMPLES)

        test = torch.stack([state1,state2],dim=0)
        true_model = lambda x: torch.tensor([-0.9*x[0]*x[1],0.8*x[0]*x[1]])
        
        pred_response = torch.stack([model.net(test[:,i]) for i in range(test.shape[1])])
        true_response = torch.stack([true_model(test[:,i]) for i in range(test.shape[1])])

        #Use a fairly focussed regressor.
        model=pysr.PySRRegressor(
            niterations=40,
            binary_operators=["+","*"],
            loss="loss(prediction,target)=(prediction-target)^2"
            )
        
        test_transpose =torch.transpose(test,0,1)

        model.fit(test_transpose,pred_response)

        err = torch.mean(torch.sqrt(torch.square(torch.tensor(model.predict(test_transpose))-true_response)))
    """

    #return [SSE, RMSE, TimeTaken, NumParams, pred_y,err]
    return [SSE, RMSE, TimeTaken, NumParams, pred_y]

def make_plots(NODE_Params, HybridParams,ensemble_size=1,NODE_FILE_NAME='TimeInvariant_NODE_Ensembles.png',HYBRID_FILE_NAME='TimeInvariant_Hybrid_ODE_Ensembles.png'):
    LV_NeuralODE_Results = []
    LV_hybridODE_Results = []

    global true_y

    #true_y #*= (1 + torch.randn(int((3/2)*data_size+1),1,2)/20.)
    #print(true_y)

    for i in range(ensemble_size):
        print(i)
        LV_NeuralODE_Results.append(LV_neuralODE_experiment(*NODE_Params))
        LV_hybridODE_Results.append(LV_hybridODE_experiment(*HybridParams))
    
    plt.figure(figsize=(12,6))
    plt.title("Pure NODE Predictions",fontsize=16)
    plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,0].cpu().numpy(), 'o',color="blue",label='Simulated Data Pop 1')
    plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,1].cpu().numpy(), 'o',color="darkorange",label='Simulated Data Pop 2')
    plt.xlabel("Time",fontsize=14)
    plt.ylabel("Population",fontsize=14)
    for i in range(ensemble_size):
        print(i)
        plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[i][4][:,0][:,0].detach().cpu().numpy(), color = "dodgerblue", alpha=0.5,label="NODE Fit Pop 1")
        plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange", alpha=0.5,label="NODE Fit Pop 2")
        #plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,0].detach().cpu().numpy(), color = "blue",alpha=0.5,label= "Hybrid ODE Fit Pop 1")
        #plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange",alpha=0.5,label= "Hybrid ODE Fit Pop 2")

    '''
    Create a vertical line.
    '''
    plt.vlines(10,0,8,color='red',linestyles='dotted')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize=14)
    
    #plt.legend()
    plt.savefig(NODE_FILE_NAME)

    plt.figure(figsize=(12,6))
    plt.title("Hybrid NODE Predictions",fontsize=16)
    plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,0].cpu().numpy(),'o', color="blue",label='Simulated Data Pop 1')
    plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,1].cpu().numpy(),'o',color="darkorange",label='Simulated Data Pop 2')
    plt.xlabel("Time",fontsize=14)
    plt.ylabel("Population",fontsize=14)
    for i in range(ensemble_size):
        print(i)
        #plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[i][4][:,0][:,0].detach().cpu().numpy(), color = "blue", alpha=0.5,label="NODE Fit Pop 1")
        #plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange", alpha=0.5,label="NODE Fit Pop 2")
        plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,0].detach().cpu().numpy(), color = "dodgerblue",alpha=0.5,label= "Hybrid ODE Fit Pop 1")
        plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange",alpha=0.5,label= "Hybrid ODE Fit Pop 2")
    '''
    Create a vertical line.
    '''
    plt.vlines(10,0,8,color='red',linestyles='dotted')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize=14)
    
    #plt.legend()
    plt.savefig(HYBRID_FILE_NAME)

def run_experiment(sizes,replicates,NODE_FILE="Experiments/NodeExperiments",HYBRID_FILE="Experiments/HybridExperiments",KNOWN_HYBRID_FILE="Experiments/KnownHybridExperiments",PURE_MECH_FILE="Experiments/PureMechExperiments"):
    niters=10000

    #node_results = []
    #hybridode_results = []
    known_params_hybridode_results = []
    #pure_mech_results = []

    counter = 0
    for dims in sizes:
        print(dims)
        for rep in range(replicates):
            counter += 1
            print(f'Simulation {counter}')
            #hybridode_result = LV_hybridODE_experiment(dims,niters,data_size,16,256)
            #node_result = LV_neuralODE_experiment(dims,niters,data_size,16,256)
            known_params_hybridode_result = LV_known_params_hybridODE_experiment(dims,niters,data_size,16,256)
            #pure_mech_result = LV_puremech_experiment(niters,data_size,16,256)

            #node_results.append([rep,node_result[0],node_result[1],node_result[2],node_result[3], 1, str(dims).replace(',',''),niters,data_size,16,256])
            known_params_hybridode_results.append([rep,known_params_hybridode_result[0],known_params_hybridode_result[1],known_params_hybridode_result[2],known_params_hybridode_result[3], 1, str(dims).replace(',',''),niters,data_size,16,256])
            #hybridode_results.append([rep,hybridode_result[0],hybridode_result[1],hybridode_result[2],hybridode_result[3], 1, str(dims).replace(',',''),niters,data_size,16,256])
            #pure_mech_results.append([rep,pure_mech_result[0],pure_mech_result[1],pure_mech_result[2],pure_mech_result[3],niters,data_size,16,256])


    #node_results_df = pd.DataFrame(node_results, columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','Hidden Layers','Layer Dimensions','niters','data size','batch time','batch size'])
    #node_results_df.to_csv(NODE_FILE+f'_{iteration_num}.csv',index=False)
    #hybridode_results_df = pd.DataFrame(hybridode_results, columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','Hidden Layers','Layer Dimensions','niters','data size','batch time','batch size'])
    #hybridode_results_df.to_csv(HYBRID_FILE+f'_{iteration_num}.csv',index=False)
    known_hybridode_results_df = pd.DataFrame(known_params_hybridode_results,columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','Hidden Layers','Layer Dimensions','niters','data size','batch time','batch size'])
    known_hybridode_results_df.to_csv(KNOWN_HYBRID_FILE+f'_{iteration_num}.csv',index=False)

    #print(len(pure_mech_results[0]))
    #print(pure_mech_results[0])

    #pure_mech_results_df = pd.DataFrame(pure_mech_results,columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','niters','data size','batch time','batch size'])
    #pure_mech_results_df.to_csv(PURE_MECH_FILE+f'_{iteration_num}.csv')

    return None

#make_plot(((20,20),1000,1000,16,256),((20,20),1000,1000,16,256),1)
#make_plot(((30,),niters,data_size,16,256),((30,),niters,data_size,16,256),1)
'''
This section investigates what impact number of iterations has on the fit at the largest number of parameters.
The assumption we make is that for lower numbers of parameters, we would see the same effect.

By considering the resulting images, we decide that 4000 or 5000 iterations work well.
'''
#for i in range(1000,11000,1000):
#    print(i)
#    make_plot(((30,),i,data_size,16,256),((30,),i,data_size,16,256),3,NODE_FILE_NAME=f'TestImages/NODE_{i}.png',HYBRID_FILE_NAME=f'TestImages/Hybrid_{i}.png')

'''
Now, we consider whether the assumption holds true for smaller parameters, and we assess the time it takes
'''
if __name__ == "__main__":
    #NUM_ITERS=5000
    layer_sizes = [(i,) for i in list(range(2,31,2))] #Consider each value in range
    begin = time.time()
    #print(LV_hybridODE_experiment((10,),NUM_ITERS,data_size,16,256,reg_param=1.0))
    #print(LV_neuralODE_experiment((10,),NUM_ITERS,data_size,16,256,reg_param=1.0))
    #make_plots(((8,),NUM_ITERS,data_size,16,256),((8,),NUM_ITERS,data_size,16,256),5,NODE_FILE_NAME=f'TestImages/NODE_ITERS_{NUM_ITERS}.png',HYBRID_FILE_NAME=f'TestImages/HYBRID_ITERS_{NUM_ITERS}.png')
    run_experiment(layer_sizes,1)

    end = time.time()
    print(f'Time Taken: {end-begin} seconds')
