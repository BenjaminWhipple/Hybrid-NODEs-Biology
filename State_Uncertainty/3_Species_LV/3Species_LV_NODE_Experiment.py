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

from Models.ThreeSpecies_LV_NODE import *

data = torch.load("3Species_LV_data.pt")

print(data.shape)

train_data_size = 91 # Training data

train_y = data[:train_data_size,:,[1,2]]
test_y = data[train_data_size:,:,[1,2]]

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
niters=2000       # training iterations
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
tf = 15.0
t = torch.linspace(t0, 9.0, train_data_size).to(device)

full_t = torch.linspace(t0, tf, 151).to(device)

#print(full_t.shape)
#print(data.shape)
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

SIZES = [5,15,25]
Hyperparameters = pd.read_csv("HyperparameterExperiments/3Species_LV_NODE_Results.csv")
summary = Hyperparameters.groupby(['Size', 'Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()

best_params = []
for size in SIZES:
    temp = Hyperparameters[Hyperparameters["Size"]==size]
    summary = temp.groupby(['Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()
    best = summary.idxmin()
    best_params.append(best)

attempts = 0
complete = False
broken = False

replicates = 2

Train = []
Test = []
Replicates = []
Size_Record = []

for size in SIZES:
    print(size)
    print(best_params[SIZES.index(size)])
    batch_time, batch_size, learning_rate, learning_rate_step, iterations = best_params[SIZES.index(size)]
    

    for replicate in range(replicates):

        model = neuralODE((3,3,[size,])).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

        start = time.time()

        #print("Starting training.")
        for it in range(1, iterations + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)
            #print(batch_y.shape)
            
            batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
            batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
            
            batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
            #batch_y_aug = torch.cat((batch_y, batch_y_constant), dim=3)
             
            pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
            
            MAE = torch.mean(torch.abs(pred_y[:,:,:,:-1] - batch_y))
            loss = MAE #+ L1_Reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (it) % 100 == 0:
                print(f'Size: {size}, Replicate: {replicate}, ','Iteration: ', it, '/', iterations)
                print(loss.item())

        print(f"Size {size}, Replicate {replicate}, Attempts: {attempts}")

        end = time.time()

        print(f'Time Taken: {end-start}')

        NumParams = get_n_params(model)

        # For augmented model
        train_y0_constant = torch.full((1,1),fill_value=0)
        train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

        pred_y = odeint(model, train_y0_aug.view(1,1,3), full_t, method='rk4').view(-1,1,3)

        train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,:-1] - train_y)))
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:-1] - train_y))))

        test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,:-1] - test_y)))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:-1] - test_y))))

        print(train_RMSE)
        print(test_RMSE)

        Train.append(train_RMSE)
        Test.append(test_RMSE)
        Replicates.append(replicate)
        Size_Record.append(size)
        
        torch.save(model,f'Experiments/3Species_LV_NODE/3Species_LV_NODE_{size}_{replicate}.pt')

df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record})

df.to_csv("Experiments/3Species_LV_NODE_Results.csv",index=False)
