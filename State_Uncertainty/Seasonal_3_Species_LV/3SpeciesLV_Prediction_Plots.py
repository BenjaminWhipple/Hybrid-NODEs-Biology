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

from Models.Seasonal_LV_NODE import *
from Models.Seasonal_LV_KnownHybrid import *
from Models.Seasonal_LV_UnknownHybrid import *

NODE_COLOR = "#2ca02c"
KNOWN_HYBRID_COLOR = "#1f77b4"
UNKNOWN_HYBRID_COLOR = "#ff7f0e"
DATA_COLOR = "darkgrey"


t0 = 0.

REPLICATES = 5 # Must be less than 50, as we only ran 50 replicates
NETWORK_SIZE = 5
# We should pull the best 30 for simplicity in plotting

NODE_RES = pd.read_csv("Experiments/Seasonal_LV_NODE_Results.csv")
NODE_RES_BEST = list(NODE_RES[NODE_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

KNOWN_HYBRID_RES = pd.read_csv("Experiments/Seasonal_LV_KnownParamHybrid_Results.csv")
KNOWN_HYBRID_RES_BEST = list(KNOWN_HYBRID_RES[KNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

UNKNOWN_HYBRID_RES = pd.read_csv("Experiments/Seasonal_LV_UnknownParamHybrid_Results.csv")
UNKNOWN_HYBRID_RES_BEST = list(UNKNOWN_HYBRID_RES[UNKNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])
print(UNKNOWN_HYBRID_RES[UNKNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Test Loss"])


print(NODE_RES_BEST)
print(KNOWN_HYBRID_RES_BEST)
print(UNKNOWN_HYBRID_RES_BEST)

t_vals = [i for i in range(16)]
#print(t_vals)

data = torch.load("Seasonal_LV_data.pt")
full_t = torch.linspace(t0, 15.0, 151).to(device)

train_y0 = data[0,:,[1,2]]


#model = neuralODE((2,2,[5]))
#model.load_state_dict(torch.load("Experiments/LV_NODE_Models/LV_NODE_5_0.pt"))
NODE_MODELS = []
NODE_PREDS = []
KNOWN_HYBRID_MODELS = []
KNOWN_HYBRID_PREDS = []
UNKNOWN_HYBRID_MODELS = []
UNKNOWN_HYBRID_PREDS = []
for i in range(REPLICATES):
    NODE_MODELS.append(torch.load(f"Experiments/Seasonal_LV_NODE/Seasonal_LV_NODE_{NETWORK_SIZE}_{NODE_RES_BEST[i]}.pt"))
    KNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/Seasonal_LV_KnownParamHybrid/Seasonal_LV_KnownParamHybrid_{NETWORK_SIZE}_{KNOWN_HYBRID_RES_BEST[i]}.pt"))
    UNKNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/Seasonal_LV_UnknownParamHybrid/Seasonal_LV_UnknownParamHybrid_{NETWORK_SIZE}_{UNKNOWN_HYBRID_RES_BEST[i]-1}.pt"))

for i in range(REPLICATES):
    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)
    pred_y = odeint(NODE_MODELS[i], train_y0_aug.view(1,1,3), full_t, method='rk4').view(-1,1,3)
    NODE_PREDS.append(pred_y)

    train_y0_constant = torch.full((1,1),fill_value=0)
    train_y0_aug = torch.cat((train_y0, train_y0_constant), dim=1)
    pred_y = odeint(KNOWN_HYBRID_MODELS[i], train_y0_aug.view(1,1,3), full_t, method='rk4').view(-1,1,3)
    KNOWN_HYBRID_PREDS.append(pred_y)

    pred_y = odeint(UNKNOWN_HYBRID_MODELS[i], train_y0_aug.view(1,1,3), full_t, method='rk4').view(-1,1,3)
    UNKNOWN_HYBRID_PREDS.append(pred_y)
#print(pred_y[:,0,1])

#print(data[:,0].cpu().numpy()[:,1])

#print(data)

fig, axs = plt.subplots(2,3,figsize=(12,6),layout='constrained',sharex="col",sharey="row")

fig.suptitle("3 Species Lotka-Volterra Model Comparisons",fontsize=14)
axs[0,0].set_title("Pure NODE")

axs[0,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,0],color=NODE_COLOR,alpha=0.2)

axs[0,0].set_ylabel("$S_2$")



axs[1,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,1],color=NODE_COLOR,alpha=0.2)

axs[1,0].set_ylabel("$S_3$")
#axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")

"""

axs[2,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[2,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,2],color=NODE_COLOR,alpha=0.2)

axs[2,0].set_ylabel("$S_3$")
#axs[1,0].set_xticks(t_vals)

axs[3,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,3],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[3,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,3],color=NODE_COLOR,alpha=0.2)

axs[3,0].set_ylabel("$S_4$")

axs[4,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,4],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[4,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,4],color=NODE_COLOR,alpha=0.2)

axs[4,0].set_ylabel("$S_5$")


axs[5,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,5],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[5,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,5],color=NODE_COLOR,alpha=0.2)

axs[5,0].set_ylabel("$S_6$")

axs[6,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,6],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[6,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,6],color=NODE_COLOR,alpha=0.2)

axs[6,0].set_ylabel("$S_7$")
"""

axs[0,1].set_title("Known Parameter Hybrid NODE")
axs[0,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,0],color=KNOWN_HYBRID_COLOR,alpha=0.2)

axs[1,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,1],color=KNOWN_HYBRID_COLOR,alpha=0.2)

#axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")
"""
axs[2,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[2,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,2],color=KNOWN_HYBRID_COLOR,alpha=0.2)

#axs[1,0].set_xticks(t_vals)


axs[3,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,3],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[3,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,3],color=KNOWN_HYBRID_COLOR,alpha=0.2)


axs[4,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,4],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[4,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,4],color=KNOWN_HYBRID_COLOR,alpha=0.2)



axs[5,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,5],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[5,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,5],color=KNOWN_HYBRID_COLOR,alpha=0.2)


axs[6,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,6],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[6,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,6],color=KNOWN_HYBRID_COLOR,alpha=0.2)
"""

axs[0,2].set_title("Unknown Parameter Hybrid NODE")
axs[0,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,0],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)


axs[1,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,1],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)

#axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")
"""
axs[2,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[2,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,2],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)

#axs[1,0].set_xticks(t_vals)

axs[3,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,3],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[3,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,3],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)


axs[4,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,4],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[4,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,4],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)



axs[5,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,5],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[5,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,5],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)


axs[6,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,6],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[6,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,6],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)
"""

axs[1,1].set_xlabel("Time")

#Draw vertical line to indicate the training/testing split.
for k in range(2):
    for l in range(3):
        axs[k,l].axvline(x=9, color='red', linestyle=':', linewidth=2)

#plt.tight_layout()
plt.savefig("Seasonal_LV_PREDICT.pdf")
