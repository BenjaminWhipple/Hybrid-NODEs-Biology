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

from Models.LV_NODE import *

data = torch.load("LV_data.pt")

train_data_size = 50 # Training data

train_y = data[:50,:,:]
test_y = data[50:,:,:]

train_y0 = train_y[0,:,:]
test_y0 = test_y[0,:,:]

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# If GPU acceleration is available

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
tf = 10.
t = torch.linspace(t0, tf, train_data_size).to(device)

full_t = torch.linspace(t0, 15.0, 76).to(device)

print(full_t.shape)
print(data.shape)
reg_param = 0.0

def get_batch(batch_time, batch_size, data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = train_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([train_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

attempts = 0
complete = False
broken = False

#replicates = 50
#sizes = [5, 10, 15, 20, 25, 30]
replicates = 50
#sizes = [5,10]
sizes = [5, 10, 15, 20, 25, 30]#, 35, 40, 45, 50]

Train = []
Test = []
Replicates = []
Size_Record = []
Attempts = []

start = time.time()
for size in sizes:
    for replicate in range(replicates):
        complete = False
        attempts = 0
        while complete == False:
            model = neuralODE((2,2,[size])).to(device)
            attempts += 1
            optimizer = optim.Adam(model.parameters(), lr=1e-1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning

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

                #'''
                if (it) % 500 == 0:
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
        print(NumParams)
        
        pred_y = odeint(model, train_y0.view(1,1,2), full_t, method='rk4').view(-1,1,2)
        #train_SSE = float(torch.sum(torch.square(pred_y[:150,:,:] - data[:150,:,:])))
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:50,:,:] - data[:50,:,:]))))
        #test_SSE = float(torch.sum(torch.square(pred_y[150:,:,:] - data[150:,:,:])))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[50:,:,:] - data[50:,:,:]))))
        
        #pred_test_y = odeint(model,test_y0.view(1,1,6), t, method='rk4').view(-1,1,6)

        #SSE = float(torch.sum(torch.square(pred_y - train_y)))
        #train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y - train_y))))

        #print(SSE)
        print(train_RMSE)

        #SSE = float(torch.sum(torch.square(pred_test_y - test_y)))
        #test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_test_y - test_y))))

        #print(SSE)
        print(test_RMSE)
        
        Train.append(train_RMSE)
        Test.append(test_RMSE)
        Replicates.append(replicate)
        Size_Record.append(size)
        Attempts.append(attempts)
        
        torch.save(model,f'Experiments/LV_NODE_Models/LV_NODE_{size}_{replicate}.pt')

df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record, "Attempts":Attempts})

df.to_csv("Experiments/LV_NODE_Results.csv",index=False)
"""
model = neuralODE((2,2,[15,])).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

start = time.time()

#print("Starting training.")
for it in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)
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

    if (it) % 100 == 0:
        print('Iteration: ', it, '/', niters)
        print('Loss: ', loss.item())
    
    #'''

end = time.time()

TimeTaken = end-start
print(f'Time Taken: {TimeTaken}')

NumParams = get_n_params(model)

pred_y = odeint(model, train_y0.view(1,1,2), full_t, method='rk4').view(-1,1,2)

train_SSE = float(torch.sum(torch.square(pred_y[:50,:,:] - data[:50,:,:])))
train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:50,:,:] - data[:50,:,:]))))

test_SSE = float(torch.sum(torch.square(pred_y[50:,:,:] - data[50:,:,:])))
test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[50:,:,:] - data[50:,:,:]))))

print(train_RMSE)
print(test_RMSE)

plt.figure(figsize=(20, 10))
plt.yscale('log')
plt.plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy(), 'o')
plt.plot(full_t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy(),alpha=0.5)
plt.savefig('LV_testing_NODE.png')

print(f'Time taken = {end-start} seconds')
#test_pred = 
"""
