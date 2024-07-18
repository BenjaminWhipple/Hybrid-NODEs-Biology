import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters for transmission, recovery, and migration
beta1, beta2 = 0.3, 0.2  # Transmission rates for each site
gamma1, gamma2 = 0.1, 0.1  # Recovery rates for each site
mu12, mu21 = 0.01, 0.01  # Migration rates between the two sites

# Initial conditions: [S1, I1, R1, S2, I2, R2] as fractions of the total population
# The sum of all these should be equal to 1
initial_conditions = [0.49, 0.01, 0, 0.50, 0.0, 0]  # Example starting values

# Time vector for the simulation
t = np.linspace(0, 160, 160)

# Normalized SIR model differential equations for the total system
def deriv(t, y):
    S1, I1, R1, S2, I2, R2 = y
    N1 = S1 + I1 + R1  # Total for site 1 (not necessarily needed but useful for clarity)
    N2 = S2 + I2 + R2  # Total for site 2
    dS1dt = -beta1 * S1 * I1 / N1 - mu12 * S1 + mu21 * S2
    dI1dt = beta1 * S1 * I1 / N1 - gamma1 * I1 - mu12 * I1 + mu21 * I2
    dR1dt = gamma1 * I1 - mu12 * R1 + mu21 * R2
    dS2dt = -beta2 * S2 * I2 / N2 - mu21 * S2 + mu12 * S1
    dI2dt = beta2 * S2 * I2 / N2 - gamma2 * I2 - mu21 * I2 + mu12 * I1
    dR2dt = gamma2 * I2 - mu21 * R2 + mu12 * R1
    return [dS1dt, dI1dt, dR1dt, dS2dt, dI2dt, dR2dt]

# Solve the differential equations across the defined timeframe
sol = solve_ivp(deriv, [t[0], t[-1]], initial_conditions, t_eval=t, method='RK45')

# Plotting the results
plt.figure(figsize=(12, 5))
plt.plot(sol.t, sol.y[0], 'b', label='Susceptible Site 1')
plt.plot(sol.t, sol.y[1], 'r', label='Infected Site 1')
plt.plot(sol.t, sol.y[2], 'g', label='Recovered Site 1')
plt.plot(sol.t, sol.y[3], 'b--', label='Susceptible Site 2')
plt.plot(sol.t, sol.y[4], 'r--', label='Infected Site 2')
plt.plot(sol.t, sol.y[5], 'g--', label='Recovered Site 2')
plt.xlabel('Time / days')
plt.ylabel('Fraction of Total Population')
plt.legend()
plt.title('Normalized Two-Site SIR Model Across Both Sites')
plt.show()

