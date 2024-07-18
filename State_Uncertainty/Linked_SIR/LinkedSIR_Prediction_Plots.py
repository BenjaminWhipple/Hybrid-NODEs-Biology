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

from Models.LinkedSIR_NODE import *
from Models.LinkedSIR_HybridModel_KnownParams import *
from Models.LinkedSIR_HybridModel_UnknownParams import *

NODE_COLOR = "#2ca02c"
KNOWN_HYBRID_COLOR = "#1f77b4"
UNKNOWN_HYBRID_COLOR = "#ff7f0e"
DATA_COLOR = "darkgrey"

t0 = 0.

REPLICATES = 10 # Must be less than 50, as we only ran 50 replicates
NETWORK_SIZE = 15
# We should pull the best 30 for simplicity in plotting

NODE_RES = pd.read_csv("Experiments/Pure_NODE_Results.csv")
NODE_RES_BEST = list(NODE_RES[NODE_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

KNOWN_HYBRID_RES = pd.read_csv("Experiments/KnownParam_Hybrid_Results.csv")
KNOWN_HYBRID_RES_BEST = list(KNOWN_HYBRID_RES[KNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

UNKNOWN_HYBRID_RES = pd.read_csv("Experiments/UnknownParam_Hybrid_Results.csv")
UNKNOWN_HYBRID_RES_BEST = list(UNKNOWN_HYBRID_RES[UNKNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

print(NODE_RES_BEST)
print(KNOWN_HYBRID_RES_BEST)
print(UNKNOWN_HYBRID_RES_BEST)

t_vals = [i for i in range(7)]
print(t_vals)

data = torch.load("LinkedSIR_test_data.pt")

data_size=100
full_t = torch.linspace(0., 100., data_size+1).to(device)

y0 = torch.tensor([[0.4,0.1, 0.0, 0.45, 0.05, 0.0]]).to(device)


#model = neuralODE((2,2,[5]))
#model.load_state_dict(torch.load("Experiments/LV_NODE_Models/LV_NODE_5_0.pt"))
NODE_MODELS = []
NODE_PREDS = []
KNOWN_HYBRID_MODELS = []
KNOWN_HYBRID_PREDS = []
UNKNOWN_HYBRID_MODELS = []
UNKNOWN_HYBRID_PREDS = []
for i in range(REPLICATES):
    NODE_MODELS.append(torch.load(f"Experiments/Pure_NODE/NODE_{NETWORK_SIZE}_{NODE_RES_BEST[i]}_data.pt"))
    KNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/KnownParam_Hybrid/KnownHybrid_{NETWORK_SIZE}_{KNOWN_HYBRID_RES_BEST[i]}_data.pt"))
    UNKNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/UnknownParam_Hybrid/UnknownHybrid_{NETWORK_SIZE}_{UNKNOWN_HYBRID_RES_BEST[i]}_data.pt"))

for i in range(REPLICATES):
    NODE_PREDS.append(odeint(NODE_MODELS[i], y0.view(1,1,6), full_t, method='rk4').view(-1,1,6))
    KNOWN_HYBRID_PREDS.append(odeint(KNOWN_HYBRID_MODELS[i], y0.view(1,1,6), full_t, method='rk4').view(-1,1,6))
    UNKNOWN_HYBRID_PREDS.append(odeint(UNKNOWN_HYBRID_MODELS[i], y0.view(1,1,6), full_t, method='rk4').view(-1,1,6))
#print(pred_y[:,0,1])

#print(data[:,0].cpu().numpy()[:,1])

fig, axs = plt.subplots(6,3,figsize=(12,16),layout='constrained',sharex="col",sharey="row")

fig.suptitle("Two Site SIR Model Comparisons",fontsize=14)
axs[0,0].set_title("Pure NODE")

axs[0,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,0],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,0],color=NODE_COLOR,alpha=0.2)

axs[0,0].set_ylabel("Site 1: Susceptible")



axs[1,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,1],color=NODE_COLOR,alpha=0.2)

axs[1,0].set_ylabel("Site 1: Infectious")
#axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")

axs[2,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,2],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[2,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,2],color=NODE_COLOR,alpha=0.2)

axs[2,0].set_ylabel("Site 1: Recovered")
#axs[1,0].set_xticks(t_vals)

axs[3,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,3],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[3,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,3],color=NODE_COLOR,alpha=0.2)

axs[3,0].set_ylabel("Site 2: Susceptible")

axs[4,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,4],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[4,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,4],color=NODE_COLOR,alpha=0.2)

axs[4,0].set_ylabel("Site 2: Infectious")


axs[5,0].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,5],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[5,0].plot(full_t,NODE_PREDS[i].detach().numpy()[:,0,5],color=NODE_COLOR,alpha=0.2)

axs[5,0].set_ylabel("Site 2: Recovered")


axs[0,1].set_title("Known Parameter Hybrid NODE")
axs[0,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,0],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,0],color=KNOWN_HYBRID_COLOR,alpha=0.2)

axs[1,1].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,1].plot(full_t,KNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,1],color=KNOWN_HYBRID_COLOR,alpha=0.2)

#axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")

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


axs[0,2].set_title("Unknown Parameter Hybrid NODE")
axs[0,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,0],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[0,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,0],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)


axs[1,2].plot(full_t.detach().cpu().numpy(), data[:,0].cpu().numpy()[:,1],'o',color=DATA_COLOR,markersize=5)
for i in range(REPLICATES):
    axs[1,2].plot(full_t,UNKNOWN_HYBRID_PREDS[i].detach().numpy()[:,0,1],color=UNKNOWN_HYBRID_COLOR,alpha=0.2)

#axs[1,0].set_xticks(t_vals)
#axs[1,0].set_xlabel("Time")

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


axs[5,1].set_xlabel("Time")

#plt.tight_layout()
plt.savefig("LINKED_SIR_PREDICT.pdf")
