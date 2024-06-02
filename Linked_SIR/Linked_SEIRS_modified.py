import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random
import torchdiffeq
import time

import pandas as pd
# Adjusted model parameters for periodic reinfection
"""
beta1, beta2 = 0.5, 0.5  # Increased transmission rates to ensure more infections
sigma1, sigma2 = 0.2, 0.2  # Rate from exposed to infectious
gamma1, gamma2 = 0.1, 0.1  # Recovery rates
xi1, xi2 = 0.1, 0.1  # Increased loss of immunity rates for quicker reinfection cycles
mu12, mu21 = 0.01, 0.01  # Migration rates
"""
# System Parameters
beta1, beta2 = 0.6, 0.3  # Site 1 has a higher transmission rate
sigma1, sigma2 = 0.3, 0.1  # Site 2 has slower progression to infectious
gamma1, gamma2 = 0.15, 0.05  # Site 1 has a higher recovery rate
xi1, xi2 = 0.05, 0.01  # Site 2 has slower immunity loss
mu12, mu21 = 0.01, 0.01  # Migration rates kept constant for simplicity

params = [beta1, beta2, sigma1, sigma2, gamma1, gamma2, xi1, xi2, mu12, mu21]

# Training parameters
niters=2000        # training iterations
data_size=40     # samples in dataset <- Reduced to 150 from 1000
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch

# Initial conditions: [S1, E1, I1, R1, S2, E2, I2, R2] as examples
initial_conditions = [0.99, 0.0, 0.01, 0, 1, 0, 0, 0]  # For demonstration purposes

# Time vector for the simulation
t = np.linspace(0, 40, data_size+1)  # Extended time to observe periodic behavior

# SEIRS model differential equations
def deriv(t, y):
    S1, E1, I1, R1, S2, E2, I2, R2 = y
    dS1dt = -beta1 * S1 * I1 / (S1 + E1 + I1 + R1) + xi1 * R1 - mu12 * S1 + mu21 * S2
    dE1dt = beta1 * S1 * I1 / (S1 + E1 + I1 + R1) - sigma1 * E1 - mu12 * E1 + mu21 * E2
    dI1dt = sigma1 * E1 - gamma1 * I1 - mu12 * I1 + mu21 * I2
    dR1dt = gamma1 * I1 - xi1 * R1 - mu12 * R1 + mu21 * R2
    dS2dt = -beta2 * S2 * I2 / (S2 + E2 + I2 + R2) + xi2 * R2 - mu21 * S2 + mu12 * S1
    dE2dt = beta2 * S2 * I2 / (S2 + E2 + I2 + R2) - sigma2 * E2 - mu21 * E2 + mu12 * E1
    dI2dt = sigma2 * E2 - gamma2 * I2 - mu21 * I2 + mu12 * I1
    dR2dt = gamma2 * I2 - xi2 * R2 - mu21 * R2 + mu12 * R1
    return [dS1dt, dE1dt, dI1dt, dR1dt, dS2dt, dE2dt, dI2dt, dR2dt]

# Solving the model
sol = solve_ivp(deriv, [t[0], t[-1]], initial_conditions, t_eval=t, method='RK45')

# Plotting
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], 'b', label='Susceptible Site 1')
plt.plot(sol.t, sol.y[1], 'y', label='Exposed Site 1')
plt.plot(sol.t, sol.y[2], 'r', label='Infected Site 1')
plt.plot(sol.t, sol.y[3], 'g', label='Recovered Site 1')
plt.xlabel('Time / days')
plt.ylabel('Number of Individuals')
plt.legend()
plt.title('SEIRS Model Dynamics for Site 1')

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[4], 'b--', label='Susceptible Site 2')
plt.plot(sol.t, sol.y[5], 'y--', label='Exposed Site 2')
plt.plot(sol.t, sol.y[6], 'r--', label='Infected Site 2')
plt.plot(sol.t, sol.y[7], 'g--', label='Recovered Site 2')
plt.xlabel('Time / days')
plt.ylabel('Number of Individuals')
plt.legend()
plt.title('SEIRS Model Dynamics for Site 2')

plt.tight_layout()
plt.show()


adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cpu')

### UTILITY FUNCTIONS ###
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

### DATA GENERATION ###

class SEIR_2_Site(nn.Module):
    def __init__(self,params):
        beta1, beta2, sigma1, sigma2, gamma1, gamma2, xi1, xi2, mu12, mu21 = params
        super(SEIR_2_Site, self).__init__()
        self.beta1 = beta1    #mM min-1
        self.beta2 = beta2   #mM-1 min-1
        self.sigma1 = sigma1     #mM min-1
        self.sigma2 = sigma2    #mM min-1
        self.gamma1 = gamma1   #mM min-1
        self.gamma2 = gamma2   #mM min-1
        self.xi1 = xi1    #mM min-1
        self.xi2 = xi2     #min-1
        self.mu12 = mu12 #min-1
        self.mu21 = mu21
       
    def forward(self, t, y):
        S1 = y.view(-1,8)[:,0]
        E1 = y.view(-1,8)[:,1]
        I1 = y.view(-1,8)[:,2]
        R1 = y.view(-1,8)[:,3]
        S2 = y.view(-1,8)[:,4]
        E2 = y.view(-1,8)[:,5]
        I2 = y.view(-1,8)[:,6]
        R2 = y.view(-1,8)[:,7]

        dS1 = -self.beta1 * S1 * I1 / (S1 + E1 + I1 + R1) + self.xi1 * R1 - self.mu12 * S1 + self.mu21 * S2
        dE1 = self.beta1 * S1 * I1 / (S1 + E1 + I1 + R1) - self.sigma1 * E1 - self.mu12 * E1 + self.mu21 * E2
        dI1 = self.sigma1 * E1 - self.gamma1 * I1 - self.mu12 * I1 + self.mu21 * I2
        dR1 = self.gamma1 * I1 - self.xi1 * R1 - self.mu12 * R1 + self.mu21 * R2
        dS2 = -self.beta2 * S2 * I2 / (S2 + E2 + I2 + R2) + self.xi2 * R2 - self.mu21 * S2 + self.mu12 * S1
        dE2 = self.beta2 * S2 * I2 / (S2 + E2 + I2 + R2) - self.sigma2 * E2 - self.mu21 * E2 + self.mu12 * E1
        dI2 = self.sigma2 * E2 - self.gamma2 * I2 - self.mu21 * I2 + self.mu12 * I1
        dR2 = self.gamma2 * I2 - self.xi2 * R2 - self.mu21 * R2 + self.mu12 * R1
            
        return torch.stack([dS1, dE1, dI1, dR1, dS2, dE2, dI2, dR2], dim=1).to(device)

# Initial condition, time span & parameters
true_y0 = torch.tensor([initial_conditions]).to(device)
t = torch.linspace(0., 80., int(2*data_size)+1).to(device)
#p = torch.tensor([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.]).to(device)


# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(SEIR_2_Site(params), true_y0, t, method='dopri5')
print("Data generated.")

print(true_y)
# Add noise (mean = 0, std = 0.1)
#true_y *= (1 + torch.randn(int((3/2)*data_size)+1,1,8)/20.)
