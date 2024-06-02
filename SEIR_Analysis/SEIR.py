import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the model parameters
beta = 0.3  # Transmission rate
sigma = 1/5  # Incubation rate (1/duration of incubation)
gamma = 1/10  # Recovery rate (1/duration of infectious period)
N = 1000  # Total population

# Define the SEIR differential equations
def seir_model(t, y, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Initial conditions
S0 = 990  # Initial susceptible population
E0 = 10   # Initial exposed population
I0 = 0    # Initial infectious population
R0 = 0    # Initial recovered population
y0 = [S0, E0, I0, R0]

# Time vector
t = np.linspace(0, 100, 1000)  # 100 days

# Solve the system of differential equations
sol = solve_ivp(seir_model, [t.min(), t.max()], y0, args=(N, beta, sigma, gamma), t_eval=t)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Susceptible')
plt.plot(sol.t, sol.y[1], label='Exposed')
plt.plot(sol.t, sol.y[2], label='Infectious')
plt.plot(sol.t, sol.y[3], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.title('SEIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()

