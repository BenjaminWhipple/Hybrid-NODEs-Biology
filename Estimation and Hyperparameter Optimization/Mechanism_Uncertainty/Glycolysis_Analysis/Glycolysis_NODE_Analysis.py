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
### COMPUTE BEST HYPERPARAMETERS
SIZES = [5,15,25]
Hyperparameters = pd.read_csv("HyperparameterExperiments/Glycolysis_NODE_Results.csv")
summary = Hyperparameters.groupby(['Size', 'Batch Time', 'Batch Size', 'Learning Rate', 'Learning Rate Step', 'Iterations'])["Train Loss"].mean()

best_params = []
for size in SIZES:
    temp = Hyperparameters[Hyperparameters["Size"]==size]
    summary = temp.groupby(['Batch Time', 'Batch Size', 'Learning Rate', 'Learning Rate Step', 'Iterations'])["Train Loss"].mean()
    best = summary.idxmin()
    best_params.append(best)

### Generate ensemble of candidate models
#complete = False

attempts = 0
complete = False
broken = False

replicates = 30

Train = []
Test = []
Replicates = []
Size_Record = []
Attempts = []

for size in SIZES:
    print(size)
    print(best_params[SIZES.index(size)])
    batch_time, batch_size, learning_rate, learning_rate_step, iterations = best_params[SIZES.index(size)]

    for replicate in range(replicates):
        complete = False
        attempts = 0
        while complete == False:
            model = neuralODE((7,7,[size])).to(device)
            attempts += 1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

            for it in range(1, iterations + 1):
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

                #'''
                if (it) % 100 == 0:
                    print(f'Size: {size}, Replicate: {replicate}, ','Iteration: ', it, '/', iterations)
                    #print(loss)
                    print(loss.item())
                    #print(type(loss.item()))
                
                if torch.isnan(loss).item()==True:
                    broken = True
                    #print(f"Current Attempt: {attempts}")
                    #print("BREAK!")
                    break
                else:
                    broken = False    
                #'''
            if broken == False:
                complete = True

        print(f"Attempts: {attempts}")

        end = time.time()

        TimeTaken = end-start
        print(f'Time Taken: {TimeTaken}')
        
        NumParams = get_n_params(model)
        print(NumParams)
        
        pred_y = odeint(model, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        #train_SSE = float(torch.sum(torch.square(pred_y[:150,:,:] - data[:150,:,:])))
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:150,:,:] - data[:150,:,:]))))
        #test_SSE = float(torch.sum(torch.square(pred_y[150:,:,:] - data[150:,:,:])))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[150:,:,:] - data[150:,:,:]))))
        
        print(train_RMSE)
        print(test_RMSE)
        
        Train.append(train_RMSE)
        Test.append(test_RMSE)
        Replicates.append(replicate)
        Size_Record.append(size)
        Attempts.append(attempts)
        
        torch.save(model,f'Experiments/Glycolysis_NODE_Models/Glycolysis_NODE_{size}_{replicate}.pt')

df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record, "Attempts":Attempts})

df.to_csv("Experiments/Glycolysis_NODE_Results.csv",index=False)