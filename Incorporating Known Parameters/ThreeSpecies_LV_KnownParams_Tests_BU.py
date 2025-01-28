import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random
import torchdiffeq
import time
import dill as pickle
import json 

from torchdiffeq import odeint

from Models.ThreeSpecies_LV.ThreeSpecies_LV_True import ThreeSpecies_LV
from Models.ThreeSpecies_LV.ThreeSpecies_LV_KnownHybrid import KnownParam_HNDE
from Models.ThreeSpecies_LV.ThreeSpecies_LV_UnknownHybrid import UnknownParam_HNDE
from Models.ThreeSpecies_LV.ThreeSpecies_LV_NODE import neuralODE

#from Models.Glycolysis.Glycolysis_KnownParam_Hybrid import Known_Params_HybridODE
#from Models.Glycolysis.Glycolysis_UnknownParam_Hybrid import Unknown_Params_HybridODE
#from Models.Glycolysis.Glycolysis_NODE import neuralODE

parser = argparse.ArgumentParser()
# Prefer hyperparams (64, 64, 0.1, 1000, 2000)

# Add arguments
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='The learning rate')
parser.add_argument('--bt', '--batch_time', type=int, default=64, help='batch time steps')
parser.add_argument('--w','--width',type=int, default=5, help='Network width')
parser.add_argument('--rep','--replicate',type=int, default=0, help='Replicate of computation')

# Parse arguments
args = parser.parse_args()

learning_rate = args.lr
batch_time = args.bt
size = args.w
file_num = args.rep


device = torch.device('cpu')
NUM_SAMPLES = 151
train_data_size = int((2/3)*NUM_SAMPLES)

true_y0 = torch.tensor([[1.0,1.0,1.0,0.0]]).to(device)
t = torch.linspace(0., 15., NUM_SAMPLES).to(device)
p = torch.tensor([0.5,1.5,3.0,1.0,1.0]).to(device)

p_true_missing = np.array([1.5,3.0,1.0])
p_guess = p_true_missing+np.sqrt(0.25*p_true_missing)*np.random.randn(len(p_true_missing)) # These correspond to roughly order of magnitude estimates

print(p_guess)

t0 = 0.
tf = 15.
#t = torch.linspace(t0, 9.0, train_data_size).to(device)

full_t = torch.linspace(t0, tf, NUM_SAMPLES).to(device)

iterations=2000

batch_size, learning_rate_step = 256, 1000 # These parameters perform well for all models.

reg_param = 0.0

MAX_ATTEMPTS = 50

mech_param_dictionary = {
    "beta":1.5,
    "gamma":3.0,
    "delta":1.0
}

def compute_param_percent_error(model, param_dictionary):
    Param_Error = 0
    errors = {}
    for name, param in model.named_parameters():
        if name in mech_param_dictionary:
            Param_Error+=torch.abs(mech_param_dictionary[name]-param)
            errors[f"{name}_percent_err"]=torch.abs(mech_param_dictionary[name]-param)/mech_param_dictionary[name]
            errors[f"{name}_fit"]=param
            errors[f"{name}_true"]=mech_param_dictionary[name]
    return Param_Error,errors

def get_batch(batch_time, batch_size, data_size, data):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = data[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([data[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

def convert_tensors(data):
    if isinstance(data, dict):
        return {key: convert_tensors(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensors(item) for item in data]
    elif isinstance(data, torch.Tensor):  # Adjust for TensorFlow if using tf.Tensor
        return data.tolist()
    else:
        return data


if __name__ == "__main__":
    print("Generating data.")
    with torch.no_grad():
        true_y = odeint(ThreeSpecies_LV(), true_y0, t, method='dopri5')
    print("Data generated.")

    # Add noise
    true_y = true_y * (1 + torch.randn(NUM_SAMPLES,1,4)/20.)

    train_y = true_y[:train_data_size,:,[1,2]]
    test_y = true_y[train_data_size:,:,[1,2]]

    train_y0 = train_y[0,:,:]
    test_y0 = test_y[0,:,:]

    filler = torch.full((1,1),fill_value=0)

    train_y0 = torch.cat((train_y0,filler),dim=1)
    test_y0 = torch.cat((test_y0,filler),dim=1)

    start = time.time()
    
    # Known parameter case

    #batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size)
    #batch_y0_constant = torch.full((256,1,1),fill_value=0.0) #For augmented NODE

    model = KnownParam_HNDE(p,(3,1,[5,])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
        
        # For augmented NODE.
        batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
        #batch_y0_T = torch.full((256,1,1),fill_value=batch_t[0])
        batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
        
        batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
        
        pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        
        #Now we are going to try to incorporate a better loss fn.
        #MAE + L1 regularization of NN params.

        #MAE = torch.mean(torch.abs(pred_y[:,:,:,:-2] - batch_y))
        #L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())]))
        #loss = MAE + L1_Reg
        loss = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))

        loss.backward()
        optimizer.step()
        #scheduler.step()

        #'''

        if (it) % 100 == 0:
            print('Iteration: ', it, '/', iterations)
            print('Loss: ', loss.item())
        
        #'''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    #NumParams = get_n_params(model)

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

    plt.figure(figsize=(20, 10))
    #plt.yscale('log')
    plt.plot(full_t.detach().cpu().numpy(), true_y[:,0].cpu().numpy()[:,[1,2]], 'o')
    plt.plot(full_t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy()[:,[0,1]],alpha=0.5, label=["Measured 1", "Measured 2"])
    plt.axvline(9.0,linestyle="dotted",color="r")
    plt.savefig('3Species_LV_testing_KnownParamHybrid.png')
    
    #### LOTS OF ISSUES WITH THIS. NOT SURE IF IT WAS ALWAYS THIS BAD. MAYBE NEED TO INCORPORATE CHECKS TO PREVENT IT FROM BECOMING NEGATIVE?
    #### -> MAYBE MODIFY INTEGRATION OF NETWORK INTO MODEL? TRY TO FIGURE THIS OUT?

    # Loose Known Parameters
    model = UnknownParam_HNDE(p_true_missing,(3,1,[5,])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
        
        # For augmented NODE.
        batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
        batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
        
        batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
        
        pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')
        MAE = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))
        Param_Error = 0
        for name, param in model.named_parameters():
            if name in mech_param_dictionary:
                Param_Error+=torch.abs(mech_param_dictionary[name]-param)

        loss = MAE + Param_Error

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (it) % 100 == 0:
            print(f'Size: {size} ','Iteration: ', it, '/', iterations)
            print(Param_Error)
            print(loss.item())


    #print(f"Attempts: {attempts}")

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

    pred_y = odeint(model, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y))))
    print(train_SSE)
    print(test_SSE)
    print(train_RMSE)
    print(test_RMSE)

    plt.figure(figsize=(20, 10))
    #plt.yscale('log')
    plt.plot(full_t.detach().cpu().numpy(), true_y[:,0].cpu().numpy()[:,[1,2]], 'o')
    plt.plot(full_t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy()[:,[0,1]],alpha=0.5, label=["Measured 1", "Measured 2"])
    plt.axvline(9.0,linestyle="dotted",color="r")
    plt.legend()
    plt.savefig('3Species_LV_testing_LooseKnownParamHybrid.png')
    
    # Unknown parameters
    model = UnknownParam_HNDE(p_guess,(3,1,[5,])).to(device)
    #model = UnknownParam_HNDE(p_true_missing,(3,1,[5,])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
        
        # For augmented NODE.
        batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
        batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
        
        batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
        
        pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')
        loss = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (it) % 100 == 0:
            print(f'Size: {size} ','Iteration: ', it, '/', iterations)
            print(loss.item())


    #print(f"Attempts: {attempts}")

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

    pred_y = odeint(model, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y))))
    print(train_SSE)
    print(test_SSE)
    print(train_RMSE)
    print(test_RMSE)

    plt.figure(figsize=(20, 10))
    #plt.yscale('log')
    plt.plot(full_t.detach().cpu().numpy(), true_y[:,0].cpu().numpy()[:,[1,2]], 'o')
    plt.plot(full_t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy()[:,[0,1]],alpha=0.5, label=["Measured 1", "Measured 2"])
    plt.axvline(9.0,linestyle="dotted",color="r")
    plt.legend()
    plt.savefig('3Species_LV_testing_UnknownParamHybrid.png')
    
    # NDE case
    model = neuralODE((3,3,[5,])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler

    start = time.time()

    #print("Starting training.")
    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
        
        batch_y0_constant = torch.full((256,1,1),fill_value=0) #For augmented NODE
        batch_y_constant = torch.full((16,256,1,1),fill_value=0)
        
        batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
         
        pred_y = odeint(model, batch_y0_aug, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
        
        MAE = torch.mean(torch.abs(pred_y[:,:,:,:-1] - batch_y))
        loss = MAE

        loss.backward()
        optimizer.step()
        scheduler.step()

        #'''

        if (it) % 100 == 0:
            print('Iteration: ', it, '/', iterations)
            print('Loss: ', loss.item())
        
        #'''

    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_aug = train_y0#torch.cat((train_y0)), train_y0_constant), dim=1)

    pred_y = odeint(model, train_y0_aug.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,:-1] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:-1] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,:-1] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:-1] - test_y))))

    print(train_RMSE)
    print(test_RMSE)

    #"""
    plt.figure(figsize=(20, 10))
    #plt.yscale('log')
    plt.plot(full_t.detach().cpu().numpy(), true_y[:,0].cpu().numpy()[:,[1,2]], 'o')
    plt.plot(full_t.detach().cpu().numpy(), pred_y[:,0].detach().cpu().numpy()[:,[0,1]],alpha=0.5)
    plt.axvline(9.0,linestyle="dotted",color="r")
    plt.savefig('3Species_LV_testing_NODE.png')

    print(f'Time taken = {end-start} seconds')
