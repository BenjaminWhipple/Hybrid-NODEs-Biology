import pandas as pd
import matplotlib.pyplot as plt

KnownHybrid_Results = pd.read_csv("Experiments/LV_KnownParamHybrid_Results.csv")
UnknownHybrid_Results = pd.read_csv("Experiments/LV_UnknownParamHybrid_Results.csv")
NODE_Results = pd.read_csv("Experiments/LV_NODE_Results.csv")

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

plt.Figure(figsize=(8,4))
plt.yscale("log")
plt.title("Median Performance of Models on Lotka-Volterra")
plt.plot(sizes,KnownHybrid_summary,'o-',label="Known Parameter Hybrid NODE")
plt.plot(sizes,UnknownHybrid_summary,'o-',label="Unknown Parameter Hybrid NODE")
plt.plot(sizes,NODE_summary,'o-',label="Pure NODE")
plt.xlabel("Neural Network Width")
plt.ylabel("Root Mean Square Error")
plt.legend()
plt.tight_layout()
plt.savefig("LinkedSIR_ModelPerformance.pdf")
