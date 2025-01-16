import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchdiffeq
import time
import sys
from torchdiffeq import odeint

from Models.LV_NODE import *
from Models.LV_KnownHybrid import *
from Models.LV_UnknownHybrid import *

NODE_COLOR = "#2ca02c"
KNOWN_HYBRID_COLOR = "#1f77b4"
UNKNOWN_HYBRID_COLOR = "#ff7f0e"
DATA_COLOR = "darkgrey"

t0 = 0.

REPLICATES = 50 # Must be less than 50, as we only ran 50 replicates

t_vals = [i for i in range(16)]
print(t_vals)

data = torch.load("LV_data.pt")
full_t = torch.linspace(t0, 15.0, 76).to(device)

train_y0 = data[0,:,:]

#model = neuralODE((2,2,[5]))
#model.load_state_dict(torch.load("Experiments/LV_NODE_Models/LV_NODE_5_0.pt"))
NODE_MODELS = []
NODE_PREDS = []
KNOWN_HYBRID_MODELS = []
KNOWN_HYBRID_PREDS = []
UNKNOWN_HYBRID_MODELS = []
UNKNOWN_HYBRID_PREDS = []

for i in range(REPLICATES): #10 replicates
    NODE_MODELS.append(torch.load(f"Experiments/LV_NODE_Models/LV_NODE_5_{i}.pt"))
    KNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/LV_KnownParamHybrid_Models/LV_KnownParamHybrid_5_{i}.pt"))
    UNKNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/LV_UnknownParamHybrid_Models/LV_UnknownParamHybrid_5_{i}.pt"))

for i in range(REPLICATES):
    NODE_PREDS.append(odeint(NODE_MODELS[i], train_y0.view(1,1,2), full_t, method='rk4').view(-1,1,2))
    KNOWN_HYBRID_PREDS.append(odeint(KNOWN_HYBRID_MODELS[i], train_y0.view(1,1,2), full_t, method='rk4').view(-1,1,2))
    UNKNOWN_HYBRID_PREDS.append(odeint(UNKNOWN_HYBRID_MODELS[i], train_y0.view(1,1,2), full_t, method='rk4').view(-1,1,2))
#print(pred_y[:,0,1])

#print(data[:,0].cpu().numpy()[:,1])

fig, axs = plt.subplots(2,3,figsize=(12,6),sharex="col",sharey="row")

fig.suptitle("Lotka-Volterra Model Comparisons",fontsize=14)
axs[0,0].set_title("Pure NODE")

axs[0,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,0],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,0],color=NODE_COLOR,alpha=0.2)

axs[0,0].set_ylabel("Prey Population")

axs[1,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,1],color=NODE_COLOR,alpha=0.2)

axs[1,0].set_ylabel("Predator Population")
axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")

axs[0,1].set_title("Known Parameter Hybrid NODE")
axs[0,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,0],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,0],color=KNOWN_HYBRID_COLOR,alpha=0.2)

axs[1,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,1],color=KNOWN_HYBRID_COLOR,alpha=0.2)

axs[1,1].set_xticks(t_vals)
axs[1,1].set_xlabel("Time")

axs[0,2].set_title("Unknown Parameter Hybrid NODE")

axs[0,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,0],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,0],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)

axs[1,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,1],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)

axs[1,2].set_xticks(t_vals)
#axs[1,2].set_xlabel("Time")

#Draw vertical line to indicate the training/testing split.
for k in range(2):
    for l in range(3):
        axs[k,l].axvline(x=10, color='red', linestyle=':', linewidth=2)

plt.tight_layout()
plt.savefig("LV_PREDICT.pdf")

