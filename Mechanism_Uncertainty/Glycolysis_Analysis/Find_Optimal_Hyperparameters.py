import pandas as pd

SIZES = [5,15,25]
Hyperparameters = pd.read_csv("HyperparameterExperiments/Glycolysis_KnownParamHybrid_Results.csv")
summary = Hyperparameters.groupby(['Size', 'Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()

best_params = []
for size in SIZES:
	temp = Hyperparameters[Hyperparameters["Size"]==size]
	summary = temp.groupby(['Batch.Time', 'Batch.Size', 'Learning.Rate', 'Learning.Rate.Step', 'Iterations'])["Train.Loss"].mean()
	best = summary.idxmin()
	best_params.append(best)

print(SIZES)
print(best_params)
