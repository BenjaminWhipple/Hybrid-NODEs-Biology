import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchdiffeq
import time
import sys

import itertools

from Models.Glycolysis_UnknownParam_Hybrid import *

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

t0 = 0.
tf = 4.
t = torch.linspace(t0, tf, train_data_size).to(device)

p0=np.array([1., 100., 1., 10., 100., 1., 10., 1., 10., 1., 0.1, 0.1, 1., 1.])



Hyperparameters = pd.read_csv("HyperparameterExperiments/Glycolysis_UnknownParamHybrid_Results.csv")
summary = Hyperparameters.groupby(['Size', 'Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()

SIZES = [5,15,25]
best_params = []
for size in SIZES:
    temp = Hyperparameters[Hyperparameters["Size"]==size]
    summary = temp.groupby(['Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()
    best = summary.idxmin()
    best_params.append(best)

def get_batch(batch_time, batch_size, data_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = train_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([train_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

batch_time, batch_size, learning_rate, learning_rate_step, iterations = best_params[SIZES.index(size)]

model = Unknown_Params_HybridODE(p0,(7,7,[size])).to(device)

for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print("Values:", param.data)

attempts += 1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=0.1) #optional learning

