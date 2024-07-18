"""
[] Incorporate checkpointing

[C] Incorporate restarting for nan values.

[] Make predictions at different data sizes (maybe 5,6,...,29,30), with 30 replicates per?
    [] Figure out how we are going to save the results.

[] Figure out how we want to present the experiments.
"""


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

from Models.LinkedSIR_HybridModel_KnownParams import *

train_y = torch.load("LinkedSIR_train_data.pt")
train_y0 = torch.tensor([[0.49, 0.01, 0.0, 0.50, 0.0, 0.0]]).to(device)

test_y0 = torch.tensor([[0.4,0.1, 0.0, 0.45, 0.05, 0.0]]).to(device)
test_y = torch.load("LinkedSIR_test_data.pt")

# Wrong p
p=[0.1,0.1,0.1,0.1]

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# If GPU acceleration is available
#gpu=0
#device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#if torch.cuda.is_available():
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Utility Fns

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

### Set up training parameters

# Training parameters
niters=1000        # training iterations
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch
reg_param = 0.0


data_size = train_y.shape[0] #Number of observations across
print(train_y.shape[0])

# We need to know t0 and tf
t0 = 0.
tf = 100.
t = torch.linspace(t0, tf, data_size).to(device)


# Batch function for training
def get_batch(batch_time, batch_size,data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = train_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([train_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y


#model = Known_Params_HybridODE((6,6,[20])).to(device)

start = time.time()

### Generate ensemble of candidate models
#complete = False

attempts = 0
complete = False
broken = False

replicates = 50
#sizes = [5, 10, 15, 20, 25, 30]
#replicates = 2
#sizes = [5,10]
sizes = [5, 10, 15, 20, 25, 30]

Train = []
Test = []
Replicates = []
Size_Record = []
Attempts = []

for size in sizes:
    for replicate in range(replicates):
        complete = False
        attempts = 0
        while complete == False:
            model = Known_Params_HybridODE(p,(6,6,[size,size])).to(device)
            attempts += 1
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) #optional learning

            for it in range(1, niters + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,data_size)
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
                    print(f'Size: {size}, Replicate: {replicate}, ','Iteration: ', it, '/', niters)
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

        pred_y = odeint(model, train_y0.view(1,1,6), t, method='rk4').view(-1,1,6)
        pred_test_y = odeint(model,test_y0.view(1,1,6), t, method='rk4').view(-1,1,6)

        #SSE = float(torch.sum(torch.square(pred_y - train_y)))
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - train_y))))

        #print(SSE)
        print(train_RMSE)

        #SSE = float(torch.sum(torch.square(pred_test_y - test_y)))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_test_y - test_y))))

        #print(SSE)
        print(test_RMSE)
        
        Train.append(train_RMSE)
        Test.append(test_RMSE)
        Replicates.append(replicate)
        Size_Record.append(size)
        Attempts.append(attempts)
        
        torch.save(model,f'Experiments/WrongParam_Hybrid/WrongHybrid_{size}_{replicate}_data.pt')

df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record, "Attempts":Attempts})

df.to_csv("Experiments/WrongParam_Hybrid_Results.csv",index=False)
"""
plt.figure(figsize=(20, 10))
plt.plot(t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy())
plt.plot(t.detach().cpu().numpy(), train_y[:,0].cpu().numpy(), 'o',alpha=0.3)
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(t.detach().cpu().numpy(), pred_test_y[:,0].detach().cpu().numpy())
plt.plot(t.detach().cpu().numpy(), test_y[:,0].cpu().numpy(), 'o',alpha=0.3)
plt.show()
"""
#print(pred_y)

