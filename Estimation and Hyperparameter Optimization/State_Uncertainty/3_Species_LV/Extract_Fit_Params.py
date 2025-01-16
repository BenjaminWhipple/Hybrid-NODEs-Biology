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


NODE_COLOR = "#2ca02c"
KNOWN_HYBRID_COLOR = "#1f77b4"
UNKNOWN_HYBRID_COLOR = "#ff7f0e"
DATA_COLOR = "darkgrey"

t0 = 0.

REPLICATES = 30 # Must be less than 50, as we only ran 50 replicates
SIZES = [5,15,25]
# We should pull the best 30 for simplicity in plotting

#model = neuralODE((2,2,[5]))
#model.load_state_dict(torch.load("Experiments/LV_NODE_Models/LV_NODE_5_0.pt"))
NODE_MODELS = []
NODE_PREDS = []
KNOWN_HYBRID_MODELS = []
KNOWN_HYBRID_PREDS = []
UNKNOWN_HYBRID_MODELS = []
UNKNOWN_HYBRID_PREDS = []

params = []
sizes = []

for NETWORK_SIZE in SIZES:
    for i in range(REPLICATES):
        temp = torch.load(f"Experiments/3Species_LV_UnknownParamHybrid/3Species_LV_UnknownParamHybrid_{NETWORK_SIZE}_{i}.pt")
        
        print(list(temp.parameters())[0].detach().numpy())
        params.append(list(temp.parameters())[0].detach().numpy())
        sizes.append(NETWORK_SIZE)

param_names = ["beta","gamma","delta"]

df = pd.DataFrame(data=np.array(params),columns=param_names)

df["Size"]=sizes

print(df)
df.to_csv("Fit_Parameters.csv",index=False)

