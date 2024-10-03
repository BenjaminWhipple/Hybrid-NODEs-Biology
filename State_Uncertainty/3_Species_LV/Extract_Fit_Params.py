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

from Models.ThreeSpecies_LV_NODE import *
from Models.ThreeSpecies_LV_KnownHybrid import *
from Models.ThreeSpecies_LV_UnknownHybrid import *

NODE_COLOR = "#2ca02c"
KNOWN_HYBRID_COLOR = "#1f77b4"
UNKNOWN_HYBRID_COLOR = "#ff7f0e"
DATA_COLOR = "darkgrey"


t0 = 0.

REPLICATES = 15 # Must be less than 50, as we only ran 50 replicates
NETWORK_SIZE = 5
# We should pull the best 30 for simplicity in plotting

NODE_RES = pd.read_csv("Experiments/3Species_LV_NODE_Results.csv")
NODE_RES_BEST = list(NODE_RES[NODE_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

KNOWN_HYBRID_RES = pd.read_csv("Experiments/3Species_LV_KnownParamHybrid_Results.csv")
KNOWN_HYBRID_RES_BEST = list(KNOWN_HYBRID_RES[KNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

UNKNOWN_HYBRID_RES = pd.read_csv("Experiments/3Species_LV_UnknownParamHybrid_Results.csv")
UNKNOWN_HYBRID_RES_BEST = list(UNKNOWN_HYBRID_RES[UNKNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])
print(UNKNOWN_HYBRID_RES[UNKNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Test Loss"])


print(NODE_RES_BEST)
print(KNOWN_HYBRID_RES_BEST)
print(UNKNOWN_HYBRID_RES_BEST)

t_vals = [i for i in range(16)]
#print(t_vals)

data = torch.load("3Species_LV_data.pt")
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
    NODE_MODELS.append(torch.load(f"Experiments/3Species_LV_NODE/3Species_LV_NODE_{NETWORK_SIZE}_{NODE_RES_BEST[i]}.pt"))
    KNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/3Species_LV_KnownParamHybrid/3Species_LV_KnownParamHybrid_{NETWORK_SIZE}_{KNOWN_HYBRID_RES_BEST[i]}.pt"))
    UNKNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/3Species_LV_UnknownParamHybrid/3Species_LV_UnknownParamHybrid_{NETWORK_SIZE}_{UNKNOWN_HYBRID_RES_BEST[i]}.pt"))
    
param_names = ["beta","gamma","delta"]

params = []
for i in range(REPLICATES):
    params.append(list(UNKNOWN_HYBRID_MODELS[i].parameters())[0].detach().numpy())

print(np.array(params))

df = pd.DataFrame(data=np.array(params),columns=param_names)
print(df)
df.to_csv("Fit_Parameters.csv",index=False)
