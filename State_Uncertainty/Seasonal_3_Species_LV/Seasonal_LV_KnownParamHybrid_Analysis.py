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

from Models.Seasonal_LV_KnownHybrid import *

data = torch.load("Seasonal_LV_data.pt")

print(data.shape)

train_data_size = 61 # Training data

train_y = data[:train_data_size,:,[1,2]]
test_y = data[train_data_size:,:,[1,2]]

train_y0 = train_y[0,:,:]
test_y0 = test_y[0,:,:]

print(train_y0)
print(test_y0)

filler = torch.full((train_y0.shape[0],train_y0.shape[1]),fill_value=0)

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
niters=5000        # training iterations
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
tf = 9.
t = torch.linspace(t0, 6.0, train_data_size).to(device)

# True p
p = torch.tensor([1.5,3.0,1.0]).to(device)

# p0 initial guess is at approximate order of magnitude.
#p0 = torch.tensor([1., 100., 1., 10., 100., 1., 10., 1., 10., 1., 0.1, 0.1, 1., 1.]).to(device)

full_t = torch.linspace(t0, 9.0, 91).to(device)

print(full_t.shape)
print(data.shape)
reg_param = 0.0

def get_batch(batch_time, batch_size, data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = train_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([train_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)

print(batch_y0.shape)
print(batch_t.shape)
print(batch_y.shape)

print(batch_y0)
print(batch_y)

print(batch_t)
print(batch_t[0])
#print(batch_t.view(16,1,1,1))
#print(batch_t.view(16,1,1,1).expand(16,256,1,1))
print(torch.full((256,1,1),fill_value=batch_t[0]))
batch_y0_constant = torch.full((256,1,1),fill_value=0.0) #For augmented NODE
print(batch_y0_constant)
#"""
# Old code

model = hybridODE(p,(4,1,[10,10])).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler

start = time.time()

#print("Starting training.")
for it in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)
    
    # For augmented NODE.
    batch_y0_constant = torch.full((256,1,1),fill_value=0) #For augmented NODE
    batch_y0_T = torch.full((256,1,1),fill_value=batch_t[0])
    batch_y_constant = torch.full((16,256,1,1),fill_value=0)
    
    batch_y0_aug = torch.cat((batch_y0, batch_y0_T, batch_y0_constant), dim=2)
    
    pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
    
    #Now we are going to try to incorporate a better loss fn.
    #MAE + L1 regularization of NN params.

    #MAE = torch.mean(torch.abs(pred_y[:,:,:,:-2] - batch_y))
    #L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())]))
    #loss = MAE + L1_Reg
    loss = torch.mean(torch.square(pred_y[:,:,:,:-2] - batch_y))

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

train_y0_constant = torch.full((1,2),fill_value=0)
train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

#print(train_y0_aug)
print(train_y0)

pred_y = odeint(model, train_y0.view(1,1,4), full_t, method='rk4').view(-1,1,4)

print(pred_y)
print(train_y)

train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,1:-2] - train_y)))
train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,1:-2] - train_y))))

test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,1:-2] - test_y)))
test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,1:-2] - test_y))))
print(train_RMSE)
print(test_RMSE)


plt.figure(figsize=(20, 10))
#plt.yscale('log')
plt.plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,[1,2]], 'o')
plt.plot(full_t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy()[:,[0,1]],alpha=0.5)
plt.axvline(6.0,linestyle="dotted",color="r")
plt.savefig('Seasonal_LV_testing_KnownParamHybrid.png')
#test_pred = 

"""
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
sizes = [5, 10, 15, 20, 25, 30]#, 35, 40, 45, 50]

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
            model = Known_Params_HybridODE(p,(7,7,[size])).to(device)
            attempts += 1
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning

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
        
        pred_y = odeint(model, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        #train_SSE = float(torch.sum(torch.square(pred_y[:150,:,:] - data[:150,:,:])))
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:150,:,:] - data[:150,:,:]))))
        #test_SSE = float(torch.sum(torch.square(pred_y[150:,:,:] - data[150:,:,:])))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[150:,:,:] - data[150:,:,:]))))
        
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
        
        torch.save(model,f'Experiments/Glycolysis_KnownParamHybrid_Models/Glycolysis_KnownParamHybrid_{size}_{replicate}.pt')

df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record, "Attempts":Attempts})

df.to_csv("Experiments/Glycolysis_KnownParamHybrid_Results.csv",index=False)
"""