'''
Notes to self:

This took about 3 minutes to run on balanced (141 seconds). Not terrible by any means.
'''

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

import pandas as pd
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
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Training parameters
niters=2000        # training iterations
data_size=150     # samples in dataset <- Reduced to 150 from 1000
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch


### UTILITY FUNCTIONS ###
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

### DATA GENERATION ###



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

# Initial condition, time span & parameters
true_y0 = torch.tensor([[1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1]]).to(device)
t = torch.linspace(0., 6., int((3/2)*data_size)+1).to(device)
p = torch.tensor([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.]).to(device)


# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(glycolysis(), true_y0, t, method='dopri5')
print("Data generated.")

# Add noise (mean = 0, std = 0.1)
true_y *= (1 + torch.randn(int((3/2)*data_size)+1,1,7)/20.)

# Batch function
def get_batch(batch_time, batch_size, data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

### MODELS ###

# Purely first-principles model based on incomplete knowledge
class pureODE(nn.Module):

    def __init__(self, p0):
        super(pureODE, self).__init__()

        self.paramsODE = nn.Parameter(p0)
        self.J0 = self.paramsODE[0]     #mM min-1
        self.k2 = self.paramsODE[2]     #mM min-1
        self.k3 = self.paramsODE[3]     #mM min-1
        self.k4 = self.paramsODE[4]     #mM min-1
        self.k5 = self.paramsODE[5]     #mM min-1
        self.k6 = self.paramsODE[6]     #mM min-1
        self.k = self.paramsODE[7]      #min-1
        self.kappa = self.paramsODE[8]  #min-1
        self.psi = self.paramsODE[11]
        self.N = self.paramsODE[12]     #mM
        self.A = self.paramsODE[13]     #mM

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0
        dS2 = - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).view(-1,1,7).to(device)

# Data-driven model
class neuralODE(nn.Module):

    def __init__(self, network_size):
        super(neuralODE, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(7, network_size),
            nn.Tanh(),
            nn.Linear(network_size, 7)
            #nn.Tanh(),
            #nn.Linear(20, 7),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.5)

    def forward(self, t, y):

        return self.net(y)

# Integrated first-principles/data-driven model
class hybridODE(nn.Module):

    def __init__(self, p0, network_size):
        super(hybridODE, self).__init__()

        self.paramsODE = p0#nn.Parameter(p0)
        self.J0 = self.paramsODE[0]     #mM min-1
        #self.k1 = self.paramsODE[1]     #mM-1 min-1
        self.k2 = self.paramsODE[2]     #mM min-1
        self.k3 = self.paramsODE[3]     #mM min-1
        self.k4 = self.paramsODE[4]     #mM min-1
        self.k5 = self.paramsODE[5]     #mM min-1
        self.k6 = self.paramsODE[6]     #mM min-1
        self.k = self.paramsODE[7]      #min-1
        self.kappa = self.paramsODE[8]  #min-1
        #self.q = self.paramsODE[9]
        #self.K1 = self.paramsODE[10]    #mM
        self.psi = self.paramsODE[11]
        self.N = self.paramsODE[12]     #mM
        self.A = self.paramsODE[13]     #mM

        self.net = nn.Sequential(
            nn.Linear(7, network_size),
            nn.Tanh(),
            nn.Linear(network_size, 7)
            #nn.Tanh(),
            #nn.Linear(20, 7),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0 + 0. * S1 #for dimensions
        dS2 = - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return (torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).view(-1,1,7) + self.net(y)).to(device)

"""
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

    if (it) % 250 == 0:
        print('Iteration: ', it, '/', niters)

end = time.time()

### VISUALIZATION ###

pred_y = odeint(model, true_y0.view(1,1,7), t, method='rk4').view(-1,1,7)

plt.figure(figsize=(20, 10))
plt.yscale('log')
plt.plot(t.detach().cpu().numpy(), true_y[:,0].cpu().numpy(), 'o')
plt.plot(t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy(),alpha=0.5)
plt.savefig('glycolysis_testing.png')

print(f'Time taken = {end-start} seconds')
"""

def Glycolysis_neuralODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size,reg_param=1.0):
    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In Neural ODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1]]).to(device)
    t = torch.linspace(0., 4., data_size).to(device)
    true_t = torch.linspace(0.,6.,int((3/2)*data_size+1))
    p = torch.tensor([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.]).to(device)

    #true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    #t = torch.linspace(0., 10., data_size).to(device)
    #true_t = torch.linspace(0.,15.,int((3/2)*data_size+1)).to(device)
    train_t = torch.linspace(0.,4.,(data_size)+1).to(device)
    #p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(glycolysis(), true_y0, true_t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    measured_y = (1 + torch.randn(int((3/2)*data_size)+1,1,2)/20.)
    train_y = measured_y[:data_size+1]
    #print(train_y)
    #print(len(train_y))

    model = neuralODE(layer_sizes[0]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler

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

    pred_y = odeint(model, true_y0.view(1,1,7), true_t, method='rk4').view(-1,1,7)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))



    #We should also conduct a PySR regression on the NN. -> Maybe not here. NODE isn't likely to represent the function anyway.
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


def Glycolysis_hybridODE_experiment(layer_sizes,niters,data_size,batch_time,batch_size,reg_param=1.0):
    print(f'Data size: {data_size}')
    niters=niters
    data_size=data_size     # samples in dataset
    batch_time = batch_time    # steps in batch
    batch_size = batch_size   # samples per batch

    print(f'In Hybrid ODE Experiment: {[layer_sizes,niters,data_size,batch_time,batch_size]}')

    true_y0 = torch.tensor([[1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1]]).to(device)
    t = torch.linspace(0., 4., data_size).to(device)
    true_t = torch.linspace(0.,6.,int((3/2)*data_size+1))
    p = torch.tensor([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.]).to(device)

    #true_y0 = torch.tensor([[5.0,5.0]]).to(device)
    #t = torch.linspace(0., 10., data_size).to(device)
    #true_t = torch.linspace(0.,15.,int((3/2)*data_size+1)).to(device)
    train_t = torch.linspace(0.,4.,(data_size)+1).to(device)
    #p = torch.tensor([1.3, 0.9, 0.8, 1.8]).to(device)

    # Disable backprop, solve system of ODEs
    #print("Generating data.")
    with torch.no_grad():
        true_y = odeint(glycolysis(), true_y0, true_t, method='dopri5')
    #print("Data generated.")

    # Add noise (mean = 0, std = 0.1)
    #measured_y = (1 + torch.randn(int((3/2)*data_size)+1,1,2)/20.)
    #train_y = measured_y[:data_size+1]
    #print(train_y)
    #print(len(train_y))

    model = hybridODE(p,layer_sizes[0]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size, data_size)
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

    pred_y = odeint(model, true_y0.view(1,1,7), true_t, method='rk4').view(-1,1,7)

    print(pred_y.shape)
    print(true_y.shape)

    SSE = float(torch.sum(torch.square(pred_y - true_y)))
    RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - true_y))))

    #We should also conduct a PySR regression on the NN. -> Again, maybe not for this one.
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

def make_plots(NODE_Params, HybridParams,ensemble_size=1,NODE_FILE_NAME='TimeInvariant_NODE_Ensembles.png',HYBRID_FILE_NAME='TimeInvariant_Hybrid_ODE_Ensembles.png'):
    LV_NeuralODE_Results = []
    LV_hybridODE_Results = []

    global true_y

    #true_y #*= (1 + torch.randn(int((3/2)*data_size+1),1,2)/20.)
    #print(true_y)

    for i in range(ensemble_size):
        print(i)
        LV_NeuralODE_Results.append(Glycolysis_neuralODE_experiment(*NODE_Params))
        LV_hybridODE_Results.append(Glycolysis_hybridODE_experiment(*HybridParams))
    
    #colors = ['red','blue','green','yellow','orange','violet','purple']
    colors = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311',
            '#009988', '#BBBBBB', '#000000']

    plt.figure(figsize=(12,6))
    plt.title("Pure NODE Predictions",fontsize=22)

    plt.yscale('log')

    #7 for the number of states.
    for i in range(7):
        plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,i].cpu().numpy(), 'o',markersize=2,color=colors[i],label=f'Simulated S{i}')
    #plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,1].cpu().numpy(), 'o',color="blue",label='Simulated Data S2')
    plt.xlabel("Time",fontsize=16)
    plt.ylabel("State Value",fontsize=16)
    
    for i in range(7):
        for j in range(ensemble_size):
            #print(i)
            plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[j][4][:,0][:,i].detach().cpu().numpy(), color = colors[i], alpha=0.5,label=f"Fitted S{i}")
            #plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange", alpha=0.5,label="NODE Fit Pop 2")
            #plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,0].detach().cpu().numpy(), color = "blue",alpha=0.5,label= "Hybrid ODE Fit Pop 1")
            #plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange",alpha=0.5,label= "Hybrid ODE Fit Pop 2")

    '''
    Create a vertical line.
    '''
    plt.vlines(4,0,4,color='black',linestyles='dotted')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.legend()
    plt.savefig(NODE_FILE_NAME, transparent=False)

    plt.figure(figsize=(12,6))
    plt.title("Hybrid NODE Predictions",fontsize=22)
    
    plt.yscale('log')
    #7 for the number of states.
    for i in range(7):
        plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,i].cpu().numpy(), 'o',markersize=2,color=colors[i],label=f'Simulated S{i}')
    #plt.plot(t.detach().cpu().numpy(), true_y[:,0][:,1].cpu().numpy(), 'o',color="blue",label='Simulated Data S2')
    plt.xlabel("Time",fontsize=16)
    plt.ylabel("State Value",fontsize=16)
    
    for i in range(7):
        for j in range(ensemble_size):
            #print(i)
            plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[j][4][:,0][:,i].detach().cpu().numpy(), color = colors[i], alpha=0.5,label=f"Fitted S{i}")
            #plt.plot(t.detach().cpu().numpy(), LV_NeuralODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange", alpha=0.5,label="NODE Fit Pop 2")
            #plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,0].detach().cpu().numpy(), color = "blue",alpha=0.5,label= "Hybrid ODE Fit Pop 1")
            #plt.plot(t.detach().cpu().numpy(), LV_hybridODE_Results[i][4][:,0][:,1].detach().cpu().numpy(), color = "orange",alpha=0.5,label= "Hybrid ODE Fit Pop 2")

    '''
    Create a vertical line.
    '''
    plt.vlines(4,0,4,color='black',linestyles='dotted')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
    plt.tight_layout()
    
    #plt.legend()
    plt.savefig(HYBRID_FILE_NAME,transparent=False)

def run_experiment(sizes,replicates,NODE_FILE="Experiments/NodeExperiments.csv",HYBRID_FILE="Experiments/HybridExperiments.csv"):
    #niters=5000
    niters=5000

    node_results = []
    hybridode_results = []
    counter = 0
    for dims in sizes:
        print(dims)
        for rep in range(replicates):
            counter += 1
            print(f'Simulation {counter}')
            hybridode_result = Glycolysis_hybridODE_experiment(dims,niters,data_size,16,256)
            node_result = Glycolysis_neuralODE_experiment(dims,niters,data_size,16,256)
            hybridode_results.append([rep,hybridode_result[0],hybridode_result[1],hybridode_result[2],hybridode_result[3], 1, str(dims).replace(',',''),niters,data_size,16,256])
            node_results.append([rep,node_result[0],node_result[1],node_result[2],node_result[3], 1, str(dims).replace(',',''),niters,data_size,16,256])

    node_results_df = pd.DataFrame(node_results, columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','Hidden Layers','Layer Dimensions','niters','data size','batch time','batch size'])
    node_results_df.to_csv(NODE_FILE,index=False)
    hybridode_results_df = pd.DataFrame(hybridode_results, columns = ['Replicate','SSE', 'RMSE', 'TimeTaken', 'NumParams','Hidden Layers','Layer Dimensions','niters','data size','batch time','batch size'])
    hybridode_results_df.to_csv(HYBRID_FILE,index=False)

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
NUM_ITERS=1000
layer_sizes = [(i,) for i in list(range(10,55,10))] #Consider each value in range

begin = time.time()
#print(Glycolysis_hybridODE_experiment((50,),NUM_ITERS,data_size,16,256,reg_param=1.0))
#print(Glycolysis_neuralODE_experiment((50,),NUM_ITERS,data_size,16,256,reg_param=1.0))

make_plots(((50,),NUM_ITERS,data_size,16,256),((50,),NUM_ITERS,data_size,16,256),1,NODE_FILE_NAME=f'TestImages/NODE_ITERS_{NUM_ITERS}.png',HYBRID_FILE_NAME=f'TestImages/HYBRID_ITERS_{NUM_ITERS}.png')
#run_experiment(layer_sizes,5)

end = time.time()
print(f'Time Taken: {end-begin} seconds')
