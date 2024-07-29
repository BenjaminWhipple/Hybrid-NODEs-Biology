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

from Models.Seasonal_LV_UnknownHybrid import *

data = torch.load("Seasonal_LV_data.pt")

print(data.shape)

train_data_size = 91 # Training data

train_y = data[:train_data_size,:,[1,2]]
test_y = data[train_data_size:,:,[1,2]]

train_y0 = train_y[0,:,:]
test_y0 = test_y[0,:,:]

print(train_y0)
print(test_y0)

filler = torch.full((1,1),fill_value=0)

train_y0 = torch.cat((train_y0,filler),dim=1)
test_y0 = torch.cat((test_y0,filler),dim=1)
print(train_y0)
print(test_y0)

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
batch_time = 32    # steps in batch
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
tf = 15.
t = torch.linspace(t0, 9.0, train_data_size).to(device)

# True p
p = torch.tensor([1.0,1.0,1.0]).to(device)

# p0 initial guess is at approximate order of magnitude.
#p0 = torch.tensor([1., 100., 1., 10., 100., 1., 10., 1., 10., 1., 0.1, 0.1, 1., 1.]).to(device)

full_t = torch.linspace(t0, 15.0, 151).to(device)

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

replicates = 50
#sizes = [5, 10, 15, 20, 25, 30]
#replicates = 2
#sizes = [15,30]
sizes = [5, 10, 15, 20, 25, 30]

# We note that we could not avoid numerical instability in training for sizes greater than 15.

Train = []
Test = []
Replicates = []
Size_Record = []


for size in sizes:
    for replicate in range(replicates):
        model = unknownhybridODE(p,(3,1,[size,size])).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

        start = time.time()

        #print("Starting training.")
        for it in range(1, niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)
            
            # For augmented NODE.
            batch_y0_constant = torch.full((256,1,1),fill_value=0) #For augmented NODE
            batch_y_constant = torch.full((16,256,1,1),fill_value=0)
            
            batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
            
            pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')
            loss = torch.sum(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (it) % 500 == 0:
                print(f'Size: {size}, Replicate: {replicate}, ','Iteration: ', it, '/', niters)
                print(loss.item())


        print(f"Attempts: {attempts}")

        end = time.time()

        TimeTaken = end-start
        print(f'Time Taken: {TimeTaken}')

        NumParams = get_n_params(model)

        train_y0_constant = torch.full((1,1),fill_value=0)
        train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

        #print(train_y0_aug)
        #print(train_y0)

        pred_y = odeint(model, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

        #print(pred_y)
        #print(train_y)

        train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y)))
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y))))

        test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y)))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y))))
        print(train_RMSE)
        print(test_RMSE)
        
        Train.append(train_RMSE)
        Test.append(test_RMSE)
        Replicates.append(replicate)
        Size_Record.append(size)
        
        torch.save(model,f'Experiments/Seasonal_LV_UnknownParamHybrid/Seasonal_LV_UnknownParamHybrid_{size}_{replicate}.pt')
        print(f'Size: {size}, Replicate: {replicate}')

df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record})

df.to_csv("Experiments/Seasonal_LV_UnknownParamHybrid_Results.csv",index=False)