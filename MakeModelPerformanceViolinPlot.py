import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Glycolysis_KnownHybrid_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Experiments/Glycolysis_KnownParamHybrid_Results.csv")[["Size","Test Loss"]].dropna()
Glycolysis_UnknownHybrid_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Experiments/Glycolysis_UnknownParamHybrid_Results.csv")[["Size","Test Loss"]].dropna()
Glycolysis_NODE_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Experiments/Glycolysis_NODE_Results.csv")[["Size","Test Loss"]].dropna()

LV_KnownHybrid_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Experiments/3Species_LV_KnownParamHybrid_Results.csv")[["Size","Test Loss"]].dropna()
LV_UnknownHybrid_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Experiments/3Species_LV_UnknownParamHybrid_Results.csv")[["Size","Test Loss"]].dropna()
LV_NODE_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Experiments/3Species_LV_NODE_Results.csv")[["Size","Test Loss"]].dropna()

sizes = Glycolysis_KnownHybrid_Results['Size'].drop_duplicates().tolist()

Glycolysis_Data = {
	"Known Hybrid":[Glycolysis_KnownHybrid_Results[Glycolysis_KnownHybrid_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes],
	"Unknown Hybrid":[Glycolysis_UnknownHybrid_Results[Glycolysis_UnknownHybrid_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes],
	"NDE":[Glycolysis_NODE_Results[Glycolysis_NODE_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes]
}

LV_Data = {
	"Known Hybrid":[LV_KnownHybrid_Results[LV_KnownHybrid_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes],
	"Unknown Hybrid":[LV_UnknownHybrid_Results[LV_UnknownHybrid_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes],
	"NDE":[LV_NODE_Results[LV_NODE_Results["Size"]==size]["Test Loss"].to_numpy() for size in sizes]
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