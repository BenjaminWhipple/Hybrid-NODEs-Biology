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

from Models.Glycolysis_NODE import *
from Models.Glycolysis_KnownParam_Hybrid import *
from Models.Glycolysis_UnknownParam_Hybrid import *

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

for NETWORK_SIZE in SIZES:
    for i in range(REPLICATES):
        NODE_MODELS.append(torch.load(f"Experiments/Glycolysis_NODE_Models/Glycolysis_NODE_{NETWORK_SIZE}_{i}.pt"))
        KNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/Glycolysis_KnownParamHybrid_Models/Glycolysis_KnownParamHybrid_{NETWORK_SIZE}_{i}.pt"))
        UNKNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/Glycolysis_UnknownParamHybrid_Models/Glycolysis_UnknownParamHybrid_{NETWORK_SIZE}_{i}.pt"))

param_names = ["J0","k1","k2","k3","k4","k5","k6","k","kappa","q","K1","psi","N","A"]

params = []

sizes = []
for NETWORK_SIZE in SIZES:
    for i in range(REPLICATES):
        sizes.append(NETWORK_SIZE)
        params.append(list(UNKNOWN_HYBRID_MODELS[i].parameters())[0].detach().numpy())

print(np.array(params))

df = pd.DataFrame(data=np.array(params),columns=param_names)

df["Size"]=sizes
print(df)
df.to_csv("Fit_Parameters.csv",index=False)

# This is how we would generate symbolic regression stuff.
print(UNKNOWN_HYBRID_MODELS[0].net(torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0])))
