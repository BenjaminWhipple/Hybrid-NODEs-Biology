import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Glycolysis_UnknownHybrid_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Fit_Parameters.csv")

LV_UnknownHybrid_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Fit_Parameters.csv")

glycolysis_param_names = [r"$J_0$",r"$k_1$",r"$k_2$",r"$k_3$",r"$k_4$",r"$k_5$",r"$k_6$",r"$k$",r"$\kappa$",r"$q$",r"$K_1$",r"$psi$",r"$N$",r"$A$"]

true_glycolysis_params = [2.5, 6.0, 16.0,100.0, 1.28,1.0, 1.8,13.0,0.1,1.0,4.0]
fit_glycolysis_param_names = ["J0","k2","k3","k4","k5","k6","k","kappa","psi","N","A"]
fit_glycolysis_param_names_plottable = [r"$J_0$",r"$k_2$",r"$k_3$",r"$k_4$",r"$k_5$",r"$k_6$",r"$k$",r"$\kappa$",r"$psi$",r"$N$",r"$A$"]

colors = ['skyblue', 'salmon', 'lightgreen']
variables = ["25","15","5"]

sizes = Glycolysis_UnknownHybrid_Results['Size'].drop_duplicates().tolist()

Glycolysis_Data = Glycolysis_UnknownHybrid_Results[fit_glycolysis_param_names+["Size"]]

fig, axs = plt.subplots(4,3, figsize=(10,6))
axs = axs.flatten()

for i in range(len(fit_glycolysis_param_names)):
    # Yes, I know this could be cleaned up.
    size5 = Glycolysis_Data[fit_glycolysis_param_names[i]][Glycolysis_Data["Size"]==5]
    #print(np.mean(Glycolysis_Data[fit_glycolysis_param_names[i]][Glycolysis_Data["Size"]==5]))
    size15 = Glycolysis_Data[fit_glycolysis_param_names[i]][Glycolysis_Data["Size"]==15]
    #print(np.mean(Glycolysis_Data[fit_glycolysis_param_names[i]][Glycolysis_Data["Size"]==15]))    
    size25 = Glycolysis_Data[fit_glycolysis_param_names[i]][Glycolysis_Data["Size"]==25]
    #print(np.mean(Glycolysis_Data[fit_glycolysis_param_names[i]][Glycolysis_Data["Size"]==25]))
    print(size5)
    print(size15)
    print(axs[i])
    positions = np.arange(len(variables))
    print(fit_glycolysis_param_names[i])

    positions = np.arange(3)
    parts1 = axs[i].violinplot(list(size5),positions=[2], vert=False,showmeans=True)#, positions=positions, vert=False, showmeans=True)
    parts2 = axs[i].violinplot(list(size15),positions=[1], vert=False,showmeans=True)#, positions=positions, vert=False, showmeans=True)
    parts3 = axs[i].violinplot(list(size25),positions=[0], vert=False,showmeans=True)#, positions=positions, vert=False, showmeans=True)
    


    # Customize the color for each group
    #for pc in parts['bodies']:
    #    pc.set_facecolor(colors)
    #    pc.set_edgecolor('black')
    #    pc.set_alpha(0.7)
    #for partname in ('cbars', 'cmins', 'cmaxes'):
    #    vp = parts[partname]
    #    vp.set_edgecolor('black')

#for i, ax in enumerate(axs):
#    print(ax)
#    positions = np.arange(len(variables))
#    print(fit_glycolysis_param_names[i])
    """
    parts = ax[i].violinplot(list(Glycolysis_Data[fit_glycolysis_param_names[i]]))#, positions=positions, vert=False, showmeans=True)
    
    # Customize the color for each group
    for pc in parts['bodies']:
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
    """

plt.savefig("test.png")

"""

Glycolysis_Data = {
	"Unknown Hybrid":[Glycolysis_UnknownHybrid_Results[Glycolysis_UnknownHybrid_Results[param]==param]["Test Loss"].to_numpy() for size in sizes],
}

LV_Data = {
	"Unknown Hybrid":[LV_UnknownHybrid_Results[LV_UnknownHybrid_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes],
}

print(Glycolysis_Data)

variables = ["25","15","5"]
groups = ["Known Hybrid","Unknown Hybrid","NDE"]
colors = ['skyblue', 'salmon', 'lightgreen']

# Create figure for the plot
fig, ax = plt.subplots(1,2, figsize=(10, 6))

# Loop over groups and plot the violin plots
for i, (group, color) in enumerate(zip(groups, colors)):
    # Calculate positions along the y-axis for each variable (spaced by group)
    positions = np.arange(len(variables)) * (len(groups) + 1) + i
    parts = ax[0].violinplot(Glycolysis_Data[group], positions=positions, vert=False, showmeans=True)
    
    # Customize the color for each group
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')

# Set y-ticks to show variable names
ax[0].set_yticks(np.arange(len(variables)) * (len(groups) + 1) + (len(groups) - 1) / 2)
ax[0].set_yticklabels(variables)

# Add labels and title
ax[0].set_xscale("log")
ax[0].set_xlabel('Root Mean Square Error')
ax[0].set_ylabel('Network Width')
ax[0].set_title('Glycolysis')

# Create a legend manually
for i, group in enumerate(groups):
    ax[0].scatter([], [], color=colors[i], label=group)
#ax[0].legend()

# Loop over groups and plot the violin plots
for i, (group, color) in enumerate(zip(groups, colors)):
    # Calculate positions along the y-axis for each variable (spaced by group)
    positions = np.arange(len(variables)) * (len(groups) + 1) + i
    parts = ax[1].violinplot(LV_Data[group], positions=positions, vert=False, showmeans=True)
    
    # Customize the color for each group
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')

# Set y-ticks to show variable names
ax[1].set_yticks(np.arange(len(variables)) * (len(groups) + 1) + (len(groups) - 1) / 2)
ax[1].set_yticklabels(variables)

# Add labels and title
ax[1].set_xscale("log")
ax[1].set_xlabel('Root Mean Square Error')
#ax[1].set_ylabel('Network Width')
ax[1].set_title('Lotka-Volterra')

# Create a legend manually
for i, group in enumerate(groups):
    ax[1].scatter([], [], color=colors[i])
#ax[1].legend()

plt.subplots_adjust(bottom=0.25)
fig.legend(loc = "outside lower center")
plt.savefig("ModelPerformances.pdf")
"""