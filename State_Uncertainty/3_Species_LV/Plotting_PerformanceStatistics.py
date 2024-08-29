import pandas as pd
import matplotlib.pyplot as plt

KnownHybrid_Results = pd.read_csv("Experiments/3Species_LV_KnownParamHybrid_Results.csv")
UnknownHybrid_Results = pd.read_csv("Experiments/3Species_LV_UnknownParamHybrid_Results.csv")
NODE_Results = pd.read_csv("Experiments/3Species_LV_NODE_Results.csv")

print(KnownHybrid_Results)
print(UnknownHybrid_Results)
print(NODE_Results)

KnownHybrid_summary = list(KnownHybrid_Results.groupby('Size')['Test Loss'].median())
UnknownHybrid_summary = list(UnknownHybrid_Results.groupby('Size')['Test Loss'].median())
NODE_summary = list(NODE_Results.groupby('Size')['Test Loss'].median())

#sizes = KnownHybrid_summary.index.tolist()
#print(list(KnownHybrid_Results['Size']))
sizes = KnownHybrid_Results['Size'].drop_duplicates().tolist()
print(sizes)

print(KnownHybrid_Results.groupby('Size')['Test Loss'].median())

print(UnknownHybrid_Results.groupby('Size')['Test Loss'].median())

print(NODE_Results.groupby('Size')['Test Loss'].median())

#plt.Figure(figsize=(8,4))
fig, ax = plt.subplots(layout='constrained')

ax.set_yscale("log")
ax.set_title("Median Performance of Models on 3 Species Lotka Volterra")
ax.plot(sizes,KnownHybrid_summary,'o-',label="Known Parameter Hybrid NODE")
ax.plot(sizes,UnknownHybrid_summary,'o-',label="Unknown Parameter Hybrid NODE")
ax.plot(sizes,NODE_summary,'o-',label="Pure NODE")
ax.set_xlabel("Neural Network Width")
ax.set_ylabel("Root Mean Square Error")
fig.legend(loc = "outside lower center")
#plt.tight_layout()
fig.savefig("3Species_LV_ModelPerformance.pdf")
