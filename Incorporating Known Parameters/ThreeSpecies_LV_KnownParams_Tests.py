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

parser = argparse.ArgumentParser()

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

    # Known parameter case
    start = time.time()
    complete = False
    broken = False
    attempts = 0

    while complete == False:
        if attempts < MAX_ATTEMPTS:
            known_strict = KnownParam_HNDE(p,(3,1,[5,])).to(device)

            optimizer = optim.Adam(known_strict.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

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
                
                pred_y = odeint(known_strict, batch_y0_aug, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
                MAE = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))
                loss = MAE

                loss.backward()
                optimizer.step()
                
                if (it) % 100 == 0:
                    print('Iteration: ', it, '/', iterations)
                    print('Loss: ', loss.item())

                if torch.isnan(loss).item()==True:
                    broken = True
                    break
                else:
                    broken = False    

            if broken == False:
                complete = True

            print()
            print(loss.item())
            print()

        else:
            complete = True
            broken = True

    print(f"Attempts: {attempts}")
    KnownParams_Strict_Attempts = attempts
    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

    pred_y = odeint(known_strict, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y))))

    print(train_RMSE)
    print(test_RMSE)

    known_param_strict_train_rmse = train_RMSE
    known_param_strict_test_rmse = test_RMSE


    # Loose Known Parameters

    start = time.time()
    complete = False
    broken = False
    attempts = 0

    while complete == False:
        if attempts < MAX_ATTEMPTS:
            known_loose = UnknownParam_HNDE(p_true_missing,(3,1,[5,])).to(device)

            optimizer = optim.Adam(known_loose.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

            start = time.time()

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                
                # For augmented NODE.
                batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
                batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
                
                batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
                
                pred_y = odeint(known_loose, batch_y0_aug, batch_t, method='rk4')
                MAE = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))
                Param_Error = 0
                for name, param in known_loose.named_parameters():
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

                if torch.isnan(loss).item()==True:
                    broken = True
                    break
                else:
                    broken = False    

            if broken == False:
                complete = True

            print()
            print(loss.item())
            print()

        else:
            complete = True
            broken = True

    print(f"Attempts: {attempts}")
    KnownParams_Loose_Attempts = attempts
    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

    pred_y = odeint(known_loose, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y))))
    print(train_SSE)
    print(test_SSE)
    print(train_RMSE)
    print(test_RMSE)

    known_param_loose_train_rmse = train_RMSE
    known_param_loose_test_rmse = test_RMSE

    
    # Unknown parameters
    start = time.time()
    complete = False
    broken = False
    attempts = 0

    while complete == False:
        if attempts < MAX_ATTEMPTS:
            unknown = UnknownParam_HNDE(p_guess,(3,1,[5,])).to(device)

            optimizer = optim.Adam(unknown.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                
                # For augmented NODE.
                batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
                batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
                
                batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
                
                pred_y = odeint(unknown, batch_y0_aug, batch_t, method='rk4')
                loss = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))

                loss.backward()
                optimizer.step()
                scheduler.step()

                if (it) % 100 == 0:
                    print(f'Size: {size} ','Iteration: ', it, '/', iterations)
                    print(loss.item())

                if torch.isnan(loss).item()==True:
                    broken = True
                    break
                else:
                    broken = False    

            if broken == False:
                complete = True

            print()
            print(loss.item())
            print()

        else:
            complete = True
            broken = True

    print(f"Attempts: {attempts}")
    UnknownParams_Attempts = attempts
    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)

    pred_y = odeint(unknown, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,[0,1]] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,[0,1]] - test_y))))
    print(train_SSE)
    print(test_SSE)
    print(train_RMSE)
    print(test_RMSE)

    unknown_param_train_rmse = train_RMSE
    unknown_param_test_rmse = test_RMSE
    
    # NDE case
    start = time.time()
    complete = False
    broken = False
    attempts = 0

    while complete == False:
        if attempts < MAX_ATTEMPTS:
            node = neuralODE((3,3,[5,])).to(device)

            optimizer = optim.Adam(node.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                
                batch_y0_constant = torch.full((256,1,1),fill_value=0) #For augmented NODE
                batch_y_constant = torch.full((16,256,1,1),fill_value=0)
                
                batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
                 
                pred_y = odeint(node, batch_y0_aug, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
                MAE = torch.mean(torch.abs(pred_y[:,:,:,:-1] - batch_y))
                loss = MAE

                loss.backward()
                optimizer.step()
                scheduler.step()

                if (it) % 100 == 0:
                    print('Iteration: ', it, '/', iterations)
                    print('Loss: ', loss.item())

                if torch.isnan(loss).item()==True:
                    broken = True
                    break
                else:
                    broken = False    

            if broken == False:
                complete = True

            print()
            print(loss.item())
            print()

        else:
            complete = True
            broken = True

    print(f"Attempts: {attempts}")
    NDE_Attempts = attempts
    end = time.time()

    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    train_y0_aug = train_y0#torch.cat((train_y0)), train_y0_constant), dim=1)

    pred_y = odeint(node, train_y0_aug.view(1,1,3), full_t, method='rk4').view(-1,1,3)

    train_SSE = float(torch.sum(torch.square(pred_y[:train_data_size,:,:-1] - train_y)))
    train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:-1] - train_y))))

    test_SSE = float(torch.sum(torch.square(pred_y[train_data_size:,:,:-1] - test_y)))
    test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:-1] - test_y))))

    print(train_RMSE)
    print(test_RMSE)
    node_train_rmse = train_RMSE
    node_test_rmse = test_RMSE

    print(known_param_strict_train_rmse)
    print(known_param_strict_test_rmse)
    print(known_param_loose_train_rmse)
    print(known_param_loose_test_rmse)
    print(unknown_param_train_rmse)
    print(unknown_param_test_rmse)
    print(node_train_rmse)
    print(node_test_rmse)

    print(KnownParams_Strict_Attempts)
    print(KnownParams_Loose_Attempts)
    print(UnknownParams_Attempts)
    print(NDE_Attempts)

    loose_param_err, loose_err_dict = compute_param_percent_error(known_loose, mech_param_dictionary)
    unknown_param_err, unknown_err_dict = compute_param_percent_error(unknown, mech_param_dictionary)

    #for k, v in err_dict.items():
    #    print(k, v)
    print(loose_param_err)
    print(unknown_param_err)

    this_data = {}
    this_data["known_param_strict_train_rmse"] = known_param_strict_train_rmse    
    this_data["known_param_strict_test_rmse"] = known_param_strict_test_rmse
    this_data["known_param_loose_train_rmse"] = known_param_loose_train_rmse
    this_data["known_param_loose_test_rmse"] = known_param_loose_test_rmse
    this_data["unknown_param_train_rmse"] = unknown_param_train_rmse
    this_data["unknown_param_test_rmse"] = unknown_param_test_rmse
    this_data["NODE_train_rmse"] = node_train_rmse
    this_data["NODE_test_rmse"] = node_test_rmse
    this_data["known_param_strict_attempts"] = KnownParams_Strict_Attempts
    this_data["known_param_loose_attempts"] = KnownParams_Loose_Attempts
    this_data["unknown_param_attempts"] = UnknownParams_Attempts
    this_data["NODE_attempts"] = NDE_Attempts
    this_data["loose_param_err"] = float(loose_param_err)
    this_data["unknown_param_err"] = float(unknown_param_err)
    this_data["loose_err_dict"] = convert_tensors(loose_err_dict)
    this_data["unknown_err_dict"] = convert_tensors(unknown_err_dict)

    with open(f'SerializedObjects/ThreeSpecies_LV_Results/data_{size}_{batch_time}_{learning_rate}_{file_num}.json', 'w') as f:
        json.dump(convert_tensors(this_data), f, indent=4)