### Boilerplate Code

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

import itertools

from Models.Glycolysis_NODE import *

data = torch.load("Glycolysis_data.pt")

train_data_size = 150 # Training data

train_y = data[:150,:,:]
test_y = data[150:,:,:]

train_y0 = train_y[0,:,:]
test_y0 = test_y[0,:,:]

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cpu')

# Training parameters
niters=2000        # training iterations
#data_size=150     # samples in dataset <- Reduced to 150 from 1000
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

t0 = 0.
tf = 4.
t = torch.linspace(t0, tf, train_data_size).to(device)

full_t = torch.linspace(t0, 6.0, 226).to(device)

print(full_t.shape)
print(data.shape)
reg_param = 0.0

def get_batch(batch_time, batch_size, data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = train_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([train_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

start = time.time()

### Generate ensemble of candidate models
#complete = False

attempts = 0
complete = False
broken = False

replicates = 9
batch_times = [16, 32, 64]
batch_sizes = [64, 128, 256]
learning_rates = [1e-1,1e-2]
learning_rate_steps = [500,1000]
iterations = [500,1000,2000,4000]
sizes =  [5, 15, 25]

print(list(itertools.product(sizes, iterations, batch_times, batch_sizes, learning_rates,learning_rate_steps)))

hyperparams = list(itertools.product(sizes, batch_times, batch_sizes, learning_rates,learning_rate_steps, iterations))

Train = []
Test = []
Replicates = []
Size_Record = []
batch_time_Record = []
batch_size_Record = []
learning_rate_Record = []
learning_rate_step_Record = []
iterations_Record = []
time_Record = []

count = 0

overall_start = time.time()
for replicate in range(replicates):
    for hyper_param in hyperparams:
        count += 1
        print()
        print(f"In parameters {count}/{len(hyperparams)*replicates}")
        
        size, batch_time, batch_size, learning_rate, learning_rate_step, niters = hyper_param

        model = neuralODE((7,7,[size,])).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

        start = time.time()

        #print("Starting training.")
        for it in range(1, niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)
            pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
            
            #Now we are going to try to incorporate a better loss fn.
            loss = torch.mean(torch.abs(pred_y - batch_y))
            MAE = torch.mean(torch.abs(pred_y - batch_y))
            
            L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())]))
            
            loss = MAE + L1_Reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (it) % 500 == 0:
                print(f'Count: {count}/{len(hyperparams)*replicates}, Size: {size}, Replicate: {replicate}, ','Iteration: ', it, '/', niters)
                print(loss.item())

        print(f"Size {size}, Replicate {replicate}, Attempts: {attempts}")

        end = time.time()

        print(f'Time Taken: {end-start}, Overall Time Taken: {end-overall_start}')
       
        pred_y = odeint(model, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:150,:,:] - data[:150,:,:]))))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[150:,:,:] - data[150:,:,:]))))
        print(train_RMSE)
        print(test_RMSE)

        Train.append(train_RMSE)
        Test.append(test_RMSE)
        Replicates.append(replicate)
        Size_Record.append(size)
        batch_time_Record.append(batch_time)
        batch_size_Record.append(batch_size)
        learning_rate_Record.append(learning_rate)
        learning_rate_step_Record.append(learning_rate_step)
        iterations_Record.append(niters)
        time_Record.append(end-start)
        
        #torch.save(model,f'HyperparameterSearch/3Species_LV_NODE/3Species_LV_NODE_{size}_{replicate}.pt')

    df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record, "Batch Time":batch_time_Record, "Batch Size":batch_size_Record, "Learning Rate":learning_rate_Record,"Learning Rate Step":learning_rate_step_Record,"Iterations":iterations_Record,"Wall Time":time_Record})

df.to_csv("HyperparameterExperiments/Glycolysis_NODE_Results.csv",index=False)
