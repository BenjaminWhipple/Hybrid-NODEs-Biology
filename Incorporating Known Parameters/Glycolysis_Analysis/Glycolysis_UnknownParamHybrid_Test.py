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

from Models.Glycolysis_UnknownParam_Hybrid import *

# Constants
NODE_COLOR = "#2ca02c"
KNOWN_HYBRID_COLOR = "#1f77b4"
UNKNOWN_HYBRID_COLOR = "#ff7f0e"
DATA_COLOR = "darkgrey"

data = torch.load("Glycolysis_data.pt")

train_data_size = 150 # Training data
full_data_size = int(1.5*train_data_size)+1 # This follows the generated data size.
train_y = data[:train_data_size,:,:]
test_y = data[train_data_size:,:,:]

train_y0 = train_y[0,:,:]
test_y0 = test_y[0,:,:]

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cpu')

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

p0=np.array([1., 100., 1., 10., 100., 1., 10., 1., 10., 1., 0.1, 0.1, 1., 1.])
full_t = torch.linspace(t0, 6.0, full_data_size).to(device)

print(full_t.shape)
print(data.shape)
reg_param = 0.0

def get_batch(batch_time, batch_size, data_size, data):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = data[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([data[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

start = time.time()

### COMPUTE BEST HYPERPARAMETERS
SIZES = [5]#,15,25]
Hyperparameters = pd.read_csv("HyperparameterExperiments/Glycolysis_UnknownParamHybrid_Results.csv")
summary = Hyperparameters.groupby(['Size', 'Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()

best_params = []
for size in SIZES:
    temp = Hyperparameters[Hyperparameters["Size"]==size]
    summary = temp.groupby(['Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()
    best = summary.idxmin()
    best_params.append(best)

### Generate ensemble of candidate models
#complete = False

attempts = 0
complete = False
broken = False

replicates = 1

Train = []
Test = []
Replicates = []
Size_Record = []
Attempts = []

for size in SIZES:
    print(size)
    print(best_params[SIZES.index(size)])
    batch_time, batch_size, learning_rate, learning_rate_step, _ = best_params[SIZES.index(size)]
    iterations = 100 # We choose to force higher iterations.

    for replicate in range(replicates):
        complete = False
        attempts = 0
        while complete == False:
            model = Unknown_Params_HybridODE(p0,(7,7,[size])).to(device)

            # Try to force a fixed parameter value.
            model.k3.requires_grad = False

            attempts += 1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                pred_y = odeint(model, batch_y0, batch_t, method='rk4') #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
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
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:] - data[:train_data_size,:,:]))))
        #test_SSE = float(torch.sum(torch.square(pred_y[150:,:,:] - data[150:,:,:])))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:] - data[train_data_size:,:,:]))))
        
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
        
        print(list(model.parameters())[0].detach().numpy())
        
        #torch.save(model,f'Experiments/Glycolysis_UnknownParamHybrid_Models/Glycolysis_UnknownParamHybrid_{size}_{replicate}.pt')

#df = pd.DataFrame({"Train Loss":Train, "Test Loss": Test, "Replicate": Replicates, "Size":Size_Record, "Attempts":Attempts})
#full_preds = odeint(model, batch_y0, full_t, method='rk4')
#df.to_csv("Experiments/Glycolysis_UnknownParamHybrid_Results.csv",index=False)

for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print("Values:", param.data)

preds = pred_y.detach().numpy()#[:,0,:]
fig, axs = plt.subplots(7,1)
for i in range(7):
    axs[i].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,i],"o",color=DATA_COLOR,markersize=1)
    axs[i].plot(full_t,preds[:,0,i],color=UNKNOWN_HYBRID_COLOR)

fig.savefig("test.png")