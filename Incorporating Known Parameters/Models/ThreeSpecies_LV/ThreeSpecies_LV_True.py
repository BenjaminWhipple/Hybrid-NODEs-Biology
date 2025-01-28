import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random
import torchdiffeq
import time

device = "cpu"

class ThreeSpecies_LV(nn.Module):
    def __init__(self):
        super(ThreeSpecies_LV, self).__init__()
        self.alpha = 0.5
        self.beta = 1.5
        self.gamma = 3.0
        self.delta = 1.0
        self.L = 1.0
        
    def forward(self, t, y):
        S1 = y.view(-1,4)[:,0]
        S2 = y.view(-1,4)[:,1]
        S3 = y.view(-1,4)[:,2]
        T = y.view(-1,4)[:,3] # We use this as a proxy for time

        dS1 = self.alpha*S1 - self.beta*S1*S2
        dS2 = self.beta*S1*S2 - self.gamma*S2*S3
        dS3 = self.gamma*S2*S3 - self.delta*S3
        dT = torch.tensor([1.0])
        return torch.stack([dS1, dS2, dS3, dT], dim=1).to(device)