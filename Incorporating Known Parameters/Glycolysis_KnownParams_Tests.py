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

from Models.Glycolysis.Glycolysis_True import glycolysis
from Models.Glycolysis.Glycolysis_KnownParam_Hybrid import Known_Params_HybridODE
from Models.Glycolysis.Glycolysis_UnknownParam_Hybrid import Unknown_Params_HybridODE
from Models.Glycolysis.Glycolysis_NODE import neuralODE

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='The learning rate')
parser.add_argument('--bt', '--batch_time', type=int, default=32, help='batch time steps')
parser.add_argument('--w','--width',type=int, default=5, help='Network width')
parser.add_argument('--rep','--replicate',type=int, default=0, help='Replicate of computation')

# Parse arguments
args = parser.parse_args()

learning_rate = args.lr
batch_time = args.bt
size = args.w
file_num = args.rep

device = torch.device('cpu')
train_data_size=150
full_data_size = int(1.5*train_data_size)+1

true_y0 = torch.tensor([[1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1]]).to(device)
t = torch.linspace(0., 6., full_data_size).to(device)
p = np.array([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.])
#p_guess = np.array([1.0, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.])
p_guess = p+np.sqrt(0.25*p)*np.random.randn(len(p)) # These correspond to roughly Order of magnitude estimates
print(p_guess)
full_t = torch.linspace(0.0, 6.0, full_data_size).to(device)

MAX_ATTEMPTS = 50

mech_param_dictionary = {
    "J0":2.5,
    "k2":6.0,
    "k3":16.0,
    "k4":100,
    "k5":1.28,
    "k6":12.0,
    "k":1.8,
    "kappa":13.0,
    "psi":0.1,
    "N":1.0,
    "A":4.0
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

iterations=2000

batch_size, learning_rate_step = 32, 500 # These parameters perform well for all models.

reg_param = 0.0


if __name__ == "__main__":
    print("Generating data.")
    with torch.no_grad():
        true_y = odeint(glycolysis(), true_y0, t, method='dopri5')
    print("Data generated.")

    # Add noise
    true_y *= (1 + torch.randn(full_data_size,1,7)/20.)

    train_data_size = 150 # Training data
    train_y = true_y[:train_data_size,:,:]
    test_y = true_y[train_data_size:,:,:]

    train_y0 = train_y[0,:,:]
    test_y0 = test_y[0,:,:]

    print("done")

    start = time.time()

    # Fit known params
    complete = False
    broken = False
    attempts = 0
    while complete == False:
        if attempts < MAX_ATTEMPTS:
            model = Known_Params_HybridODE(p,(7,7,[size,])).to(device)
            attempts += 1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
                #Now we are going to try to incorporate a better loss fn.
                MAE = torch.mean(torch.abs(pred_y - batch_y))
                #L1_Reg = reg_param*torch.sum(torch.tensor([torch.sum(torch.abs(i)) for i in list(model.parameters())]))
                loss = MAE #+ L1_Reg

                loss.backward()
                optimizer.step()
                scheduler.step()

                #'''
                if (it) % 100 == 0:
                    print(f'Size: {size}, ','Iteration: ', it, '/', iterations)
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
    print()

    if (broken != True) & (complete != False):
        pred_y = odeint(model, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:] - train_y))))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:] - test_y))))

    else:
        train_RMSE = np.nan
        test_RMSE = np.nan

    known_param_strict_train_rmse = train_RMSE
    known_param_strict_test_rmse = test_RMSE
    
    # Fit unknown params with loss.
    complete = False
    broken = False
    attempts = 0
    while complete == False:
        if attempts < MAX_ATTEMPTS:
            unknown_model_loose = Unknown_Params_HybridODE(p,(7,7,[size,])).to(device)
            attempts += 1
            optimizer = optim.Adam(unknown_model_loose.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                pred_y = odeint(unknown_model_loose, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
                #Now we are going to try to incorporate a better loss fn.
                #loss = torch.mean(torch.abs(pred_y - batch_y))
                MAE = torch.mean(torch.abs(pred_y - batch_y))
                Param_Error = 0
                for name, param in unknown_model_loose.named_parameters():
                    if name in mech_param_dictionary:
                        Param_Error+=torch.abs(mech_param_dictionary[name]-param)

                loss = MAE + Param_Error
                
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (it) % 100 == 0:
                    print(f'Size: {size}, ','Iteration: ', it, '/', iterations)
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
    KnownParams_Loose_attempts = attempts

    end = time.time()
    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    if (broken != True) & (complete != False):
        pred_y = odeint(unknown_model_loose, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:] - train_y))))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:] - test_y))))

    else:
        train_RMSE = np.nan
        test_RMSE = np.nan

    known_param_loose_train_rmse = train_RMSE
    known_param_loose_test_rmse = test_RMSE


    # Fit unknown parameters
        # Fit unknown params with loss.
    complete = False
    broken = False
    attempts = 0
    
    while complete == False:
        if attempts < MAX_ATTEMPTS:
            unknown_model = Unknown_Params_HybridODE(p_guess,(7,7,[size,])).to(device)
            attempts += 1
            optimizer = optim.Adam(unknown_model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                pred_y = odeint(unknown_model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
                #Now we are going to try to incorporate a better loss fn.
                #loss = torch.mean(torch.abs(pred_y - batch_y))
                MAE = torch.mean(torch.abs(pred_y - batch_y))
                
                loss = MAE

                loss.backward()
                optimizer.step()
                scheduler.step()

                if (it) % 100 == 0:
                    print(f'Size: {size}, ','Iteration: ', it, '/', iterations)
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

    if (broken != True) & (complete != False):
        pred_y = odeint(unknown_model, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:] - train_y))))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:] - test_y))))

    else:
        train_RMSE = np.nan
        test_RMSE = np.nan

    unknown_param_train_rmse = train_RMSE
    unknown_param_test_rmse = test_RMSE

    # Fit NDE.

    complete = False
    broken = False
    attempts = 0

    while complete == False:
        if attempts < MAX_ATTEMPTS:
            NODE = neuralODE((7,7,[size,])).to(device)
            attempts += 1
            optimizer = optim.Adam(NODE.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

            for it in range(1, iterations + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size,train_data_size,train_y)
                pred_y = odeint(NODE, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
                
                #Now we are going to try to incorporate a better loss fn.
                #loss = torch.mean(torch.abs(pred_y - batch_y))
                MAE = torch.mean(torch.abs(pred_y - batch_y))
                
                loss = MAE

                loss.backward()
                optimizer.step()
                scheduler.step()

                if (it) % 100 == 0:
                    print(f'Size: {size}, ','Iteration: ', it, '/', iterations)
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
    NODE_attempts = attempts

    end = time.time()
    TimeTaken = end-start
    print(f'Time Taken: {TimeTaken}')

    if (broken != True) & (complete != False):
        pred_y = odeint(NODE, train_y0.view(1,1,7), full_t, method='rk4').view(-1,1,7)
        train_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[:train_data_size,:,:] - train_y))))
        test_RMSE = float(torch.sqrt(torch.mean(torch.square(pred_y[train_data_size:,:,:] - test_y))))

    else:
        train_RMSE = np.nan
        test_RMSE = np.nan

    NODE_train_rmse = train_RMSE
    NODE_test_rmse = test_RMSE

    # Data serialization.

    print(known_param_strict_train_rmse)
    print(known_param_strict_test_rmse)
    print(known_param_loose_train_rmse)
    print(known_param_loose_test_rmse)
    print(unknown_param_train_rmse)
    print(unknown_param_test_rmse)
    print(NODE_train_rmse)
    print(NODE_test_rmse)

    print(KnownParams_Strict_Attempts)
    print(KnownParams_Loose_attempts)
    print(UnknownParams_Attempts)
    print(NODE_attempts)

    loose_param_err, loose_err_dict = compute_param_percent_error(unknown_model_loose, mech_param_dictionary)
    unknown_param_err, unknown_err_dict = compute_param_percent_error(unknown_model, mech_param_dictionary)

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
    this_data["NODE_train_rmse"] = NODE_train_rmse
    this_data["NODE_test_rmse"] = NODE_test_rmse
    this_data["known_param_strict_attempts"] = KnownParams_Strict_Attempts
    this_data["known_param_loose_attempts"] = KnownParams_Loose_attempts
    this_data["unknown_param_attempts"] = UnknownParams_Attempts
    this_data["NODE_attempts"] = NODE_attempts
    this_data["loose_param_err"] = float(loose_param_err)
    this_data["unknown_param_err"] = float(unknown_param_err)
    this_data["loose_err_dict"] = convert_tensors(loose_err_dict)
    this_data["unknown_err_dict"] = convert_tensors(unknown_err_dict)

    with open(f'SerializedObjects/Glycolysis_Results/data_{size}_{batch_time}_{learning_rate}_{file_num}.json', 'w') as f:
        json.dump(convert_tensors(this_data), f, indent=4)