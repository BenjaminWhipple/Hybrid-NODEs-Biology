import pandas as pd
import matplotlib.pyplot as plt

Glycolysis_KnownHybrid_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Experiments/Glycolysis_KnownParamHybrid_Results.csv")
Glycolysis_UnknownHybrid_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Experiments/Glycolysis_UnknownParamHybrid_Results.csv")
Glycolysis_NODE_Results = pd.read_csv("Mechanism_Uncertainty/Glycolysis_Analysis/Experiments/Glycolysis_NODE_Results.csv")

Glycolysis_KnownHybrid_summary = list(Glycolysis_KnownHybrid_Results.groupby('Size')['Test Loss'].median())
Glycolysis_UnknownHybrid_summary = list(Glycolysis_UnknownHybrid_Results.groupby('Size')['Test Loss'].median())
Glycolysis_NODE_summary = list(Glycolysis_NODE_Results.groupby('Size')['Test Loss'].median())

sizes = Glycolysis_KnownHybrid_Results['Size'].drop_duplicates().tolist()

LV_KnownHybrid_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Experiments/3Species_LV_KnownParamHybrid_Results.csv")
LV_UnknownHybrid_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Experiments/3Species_LV_UnknownParamHybrid_Results.csv")
LV_NODE_Results = pd.read_csv("State_Uncertainty/3_Species_LV/Experiments/3Species_LV_NODE_Results.csv")

LV_KnownHybrid_summary = list(LV_KnownHybrid_Results.groupby('Size')['Test Loss'].median())
LV_UnknownHybrid_summary = list(LV_UnknownHybrid_Results.groupby('Size')['Test Loss'].median())
LV_NODE_summary = list(LV_NODE_Results.groupby('Size')['Test Loss'].median())

fig, ax = plt.subplots(1,2, layout='constrained',figsize = (10,4))

ax[0].set_yscale("log")
ax[0].set_title("Glycolysis")
ax[0].plot(sizes,Glycolysis_KnownHybrid_summary,'o-',label="Known Parameter Hybrid NODE")
ax[0].plot(sizes,Glycolysis_UnknownHybrid_summary,'o-',label="Unknown Parameter Hybrid NODE")
ax[0].plot(sizes,Glycolysis_NODE_summary,'o-',label="Pure NODE")
ax[0].set_xticks(sizes)
ax[0].set_xlabel("Neural Network Width")
ax[0].set_ylabel("Root Mean Square Error")

ax[1].set_yscale("log")
ax[1].set_title("3 Species Lotka-Volterra")
ax[1].plot(sizes,LV_KnownHybrid_summary,'o-')#,label="Known Parameter Hybrid NODE")
ax[1].plot(sizes,LV_UnknownHybrid_summary,'o-')#,label="Unknown Parameter Hybrid NODE")
ax[1].plot(sizes,LV_NODE_summary,'o-')#,label="Pure NODE")
ax[1].set_xticks(sizes)
ax[1].set_xlabel("Neural Network Width")
#ax[1].set_ylabel("Root Mean Square Error")


fig.legend(loc = "outside lower center")
#plt.tight_layout()
fig.savefig("ModelPerformances.pdf")



