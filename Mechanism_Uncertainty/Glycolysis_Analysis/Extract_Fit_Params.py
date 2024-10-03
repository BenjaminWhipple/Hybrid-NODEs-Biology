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

REPLICATES = 15 # Must be less than 50, as we only ran 50 replicates
NETWORK_SIZE = 15
# We should pull the best 30 for simplicity in plotting

NODE_RES = pd.read_csv("Experiments/Glycolysis_NODE_Results.csv")
NODE_RES_BEST = list(NODE_RES[NODE_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

KNOWN_HYBRID_RES = pd.read_csv("Experiments/Glycolysis_KnownParamHybrid_Results.csv")
KNOWN_HYBRID_RES_BEST = list(KNOWN_HYBRID_RES[KNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])

UNKNOWN_HYBRID_RES = pd.read_csv("Experiments/Glycolysis_UnknownParamHybrid_Results.csv")
UNKNOWN_HYBRID_RES_BEST = list(UNKNOWN_HYBRID_RES[UNKNOWN_HYBRID_RES["Size"]==NETWORK_SIZE].nsmallest(REPLICATES,"Test Loss")["Replicate"])


#model = neuralODE((2,2,[5]))
#model.load_state_dict(torch.load("Experiments/LV_NODE_Models/LV_NODE_5_0.pt"))
NODE_MODELS = []
NODE_PREDS = []
KNOWN_HYBRID_MODELS = []
KNOWN_HYBRID_PREDS = []
UNKNOWN_HYBRID_MODELS = []
UNKNOWN_HYBRID_PREDS = []
for i in range(REPLICATES):
    NODE_MODELS.append(torch.load(f"Experiments/Glycolysis_NODE_Models/Glycolysis_NODE_{NETWORK_SIZE}_{NODE_RES_BEST[i]}.pt"))
    KNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/Glycolysis_KnownParamHybrid_Models/Glycolysis_KnownParamHybrid_{NETWORK_SIZE}_{KNOWN_HYBRID_RES_BEST[i]}.pt"))
    UNKNOWN_HYBRID_MODELS.append(torch.load(f"Experiments/Glycolysis_UnknownParamHybrid_Models/Glycolysis_UnknownParamHybrid_{NETWORK_SIZE}_{UNKNOWN_HYBRID_RES_BEST[i]}.pt"))

param_names = ["J0","k1","k2","k3","k4","k5","k6","k","kappa","q","K1","psi","N","A"]

params = []
for i in range(REPLICATES):
    params.append(list(UNKNOWN_HYBRID_MODELS[i].parameters())[0].detach().numpy())

print(np.array(params))

df = pd.DataFrame(data=np.array(params),columns=param_names)
print(df)
df.to_csv("Fit_Parameters.csv",index=False)
