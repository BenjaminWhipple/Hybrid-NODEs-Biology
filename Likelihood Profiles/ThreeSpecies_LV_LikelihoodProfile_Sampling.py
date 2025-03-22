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
from Models.ThreeSpecies_LV.ThreeSpecies_LV_UnknownHybrid import UnknownParam_HNDE

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='The learning rate')
parser.add_argument('--bt', '--batch_time', type=int, default=64, help='batch time steps')
parser.add_argument('--w','--width',type=int, default=5, help='Network width')
parser.add_argument('--rep','--replicate',type=int, default=0, help='Replicate of computation')
parser.add_argument('--reg','--regularization',type=float, default=0.0, help='L1 Regularization')
parser.add_argument('--pi','--parameter_index',type=int, default=0, help='Parameter index to hold fixed')
# Parse arguments
args = parser.parse_args()

learning_rate = args.lr
batch_time = args.bt
size = args.w
file_num = args.rep
regularization = args.reg
parameter_index = args.pi


device = torch.device('cpu')
NUM_SAMPLES = 151
train_data_size = int((2/3)*NUM_SAMPLES)

true_y0 = torch.tensor([[1.0,1.0,1.0,0.0]]).to(device)
t = torch.linspace(0., 15., NUM_SAMPLES).to(device)
p = torch.tensor([0.5,1.5,3.0,1.0,1.0]).to(device)

p_true_missing = np.array([1.5,3.0,1.0])
#p_guess = p_true_missing+np.sqrt(0.25*p_true_missing)*np.random.randn(len(p_true_missing))
p_guess = np.array([1.0,1.0,1.0])
print(p_guess)

t0 = 0.
tf = 15.

full_t = torch.linspace(t0, tf, NUM_SAMPLES).to(device)

MAX_ATTEMPTS = 50
PARAM_SAMPLES = 11

iterations=1000

batch_size, learning_rate_step = 256, 1000 # These parameters perform well for all models.

reg_param = regularization

MAX_ATTEMPTS = 50

mech_param_dictionary = {
    "beta":1.5,
    "gamma":3.0,
    "delta":1.0
}

mech_params = ["beta","gamma","delta"]
param_ranges = [[np.log10(mech_param_dictionary[param])-1.5,np.log10(mech_param_dictionary[param])+1.5] for param in mech_params]
this_param_range = param_ranges[parameter_index]
current_param_values = np.linspace(this_param_range[0],this_param_range[1],PARAM_SAMPLES)



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
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    else:
        return data


if __name__ == "__main__":
    for sample in range(PARAM_SAMPLES):
        # Add noise
        print("Generating data.")
        with torch.no_grad():
            true_y = odeint(ThreeSpecies_LV(), true_y0, t, method='dopri5')
        print("Data generated.")
        true_y = true_y * (1 + torch.randn(NUM_SAMPLES,1,4)/20.)

        train_y = true_y[:train_data_size,:,[1,2]]
        test_y = true_y[train_data_size:,:,[1,2]]

        train_y0 = train_y[0,:,:]
        test_y0 = test_y[0,:,:]

        filler = torch.full((1,1),fill_value=0)

        train_y0 = torch.cat((train_y0,filler),dim=1)
        test_y0 = torch.cat((test_y0,filler),dim=1)

        # Unknown parameters
        start = time.time()
        complete = False
        broken = False
        attempts = 0

        while complete == False and attempts < 50:
            attempts += 1
            print(attempts)
            if attempts < MAX_ATTEMPTS:
                unknown_model = UnknownParam_HNDE(p_guess,(3,1,[5,])).to(device)

                with torch.no_grad():
                    for name, param in unknown_model.named_parameters():
                        if name == mech_params[parameter_index]:
                            param.requires_grad = False
                            print(10**current_param_values[sample])
                            param.data = torch.tensor(10**current_param_values[sample],requires_grad=False)
                            #unknown_model.named_parameters[name]=10**current_param_values[0]

                optimizer = optim.Adam(unknown_model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning rate scheduler

                for it in range(1, iterations + 1):
                    optimizer.zero_grad()
                    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                    
                    # For augmented NODE.
                    batch_y0_constant = torch.full((batch_size,1,1),fill_value=0) #For augmented NODE
                    batch_y_constant = torch.full((batch_time,batch_size,1,1),fill_value=0)
                    
                    batch_y0_aug = torch.cat((batch_y0, batch_y0_constant), dim=2)
                    
                    pred_y = odeint(unknown_model, batch_y0_aug, batch_t, method='rk4')
                    #loss = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))
                    MAE = torch.mean(torch.abs(pred_y[:,:,:,[0,1]] - batch_y))
                    L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(unknown_model.parameters())]))
                    
                    loss = MAE + L1_Reg

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

        pred_y = odeint(unknown_model, train_y0.view(1,1,3), full_t, method='rk4').view(-1,1,3)

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
        
        print(unknown_param_train_rmse)
        print(unknown_param_test_rmse)
        print(UnknownParams_Attempts)
        
        #loose_param_err, loose_err_dict = compute_param_percent_error(unknown_model_loose, mech_param_dictionary)
        unknown_param_err, unknown_err_dict = compute_param_percent_error(unknown_model, mech_param_dictionary)

        #for k, v in err_dict.items():
        #    print(k, v)
        #print(loose_param_err)
        print(unknown_param_err)

        this_data = {}
        this_data["unknown_param_train_rmse"] = unknown_param_train_rmse
        this_data["unknown_param_test_rmse"] = unknown_param_test_rmse
        this_data["unknown_param_attempts"] = UnknownParams_Attempts
        this_data["unknown_param_err"] = float(unknown_param_err)
        this_data["unknown_err_dict"] = convert_tensors(unknown_err_dict)

        with open(f'SerializedObjects/ThreeSpecies_LV_Results/data_{size}_{batch_time}_{learning_rate}_{reg_param}_{parameter_index}_{sample}_{file_num}.json', 'w') as f:
            json.dump(convert_tensors(this_data), f, indent=4)
