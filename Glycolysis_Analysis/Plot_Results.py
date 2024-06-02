import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

NeuralODE_Data = pd.read_csv("Experiments/NodeMaster.csv")
HybridODE_Data = pd.read_csv("Experiments/HybridMaster.csv")

print(NeuralODE_Data)
print(HybridODE_Data)
print(NeuralODE_Data.columns)

print(NeuralODE_Data.groupby('Layer Dimensions'))
print(NeuralODE_Data.groupby('Layer Dimensions').mean()['RMSE'])
print(NeuralODE_Data.groupby('Layer Dimensions').var()['RMSE'])


print(HybridODE_Data.groupby('Layer Dimensions'))
print(HybridODE_Data.groupby('Layer Dimensions').mean()['RMSE'])
print(HybridODE_Data.groupby('Layer Dimensions').var()['RMSE'])

HybridODE_Summary = pd.DataFrame({})
NeuralODE_Summary = pd.DataFrame({})

HybridODE_Summary['Mean RMSE']=HybridODE_Data.groupby('Layer Dimensions').mean()['RMSE']
HybridODE_Summary['Var RMSE']=HybridODE_Data.groupby('Layer Dimensions').var()['RMSE']
print(HybridODE_Summary)

NeuralODE_Summary['Mean RMSE']=NeuralODE_Data.groupby('Layer Dimensions').mean()['RMSE']
NeuralODE_Summary['Var RMSE']=NeuralODE_Data.groupby('Layer Dimensions').var()['RMSE']
print(NeuralODE_Summary)

dims = list(NeuralODE_Data['Layer Dimensions'])
#print(dims)

strip_parens = lambda x: int(x.replace('(','').replace(')',''))

dims = [int(i.replace('(','').replace(')','')) for i in dims]
#print(dims)


unique_dims = []
for i in dims:
	if i not in unique_dims:
		unique_dims.append(i)
	else:
		pass
print(unique_dims)

#print(NeuralODE_Data['Layer Dimensions'].apply(strip_parens))
#print(HybridODE_Data['Layer Dimensions'])
print(NeuralODE_Data['Layer Dimensions'])
NeuralODE_Data['Layer Dimensions'] = NeuralODE_Data['Layer Dimensions'].apply(strip_parens)#,columns=['Layer Dimensions'])
print(NeuralODE_Data['Layer Dimensions'])

print(HybridODE_Data['Layer Dimensions'])
HybridODE_Data['Layer Dimensions'] = HybridODE_Data['Layer Dimensions'].apply(strip_parens)#,columns=['Layer Dimensions'])
print(HybridODE_Data['Layer Dimensions'])

hybrid_means = []
hybrid_vars = []
hybrid_err_bars = []

node_means = []
node_vars = []
node_err_bars = []


for i in unique_dims:
	temp_hybrid = HybridODE_Data[HybridODE_Data['Layer Dimensions'] == i] #= statistics.mean(HybridODE_Data)
	temp_node = NeuralODE_Data[NeuralODE_Data['Layer Dimensions'] == i]

	n_obs_hybrid = temp_hybrid.shape[1]
	n_obs_node = temp_hybrid.shape[1]

	print(i)
	print(statistics.mean(list(temp_hybrid['RMSE'])))
	hybrid_means.append(statistics.mean(list(temp_hybrid['RMSE'])))
	hybrid_err_bars.append(1.96*statistics.stdev(list(temp_hybrid['RMSE']))/np.sqrt(n_obs_hybrid))

	print(statistics.mean(list(temp_node['RMSE'])))
	node_means.append(statistics.mean(list(temp_node['RMSE'])))
	node_err_bars.append(1.96*statistics.stdev(list(temp_node['RMSE']))/np.sqrt(n_obs_node))

	print()

plt.figure(figsize=(8,6))

plt.title('RMSE: Glycolysis',fontsize=22)
plt.errorbar(unique_dims, node_means, yerr=node_err_bars,label="Pure NODE")
plt.errorbar(unique_dims, hybrid_means,yerr=hybrid_err_bars,label="Hybrid NODE")

plt.xlabel('Network Size',fontsize=16)
plt.ylabel('Mean RMSE',fontsize=16)
plt.legend(fontsize=14)
plt.savefig("MeanRMSE_Plots_Glycolysis.png")


###
# NOW FOR THE MECHANISTIC ERROR
###
"""
hybrid_mech_err_means = []
node_mech_err_means = []

for i in unique_dims:
	temp_hybrid = HybridODE_Data[HybridODE_Data['Layer Dimensions'] == i] #= statistics.mean(HybridODE_Data)
	temp_node = NeuralODE_Data[NeuralODE_Data['Layer Dimensions'] == i]

	print(i)
	print(statistics.mean(list(temp_hybrid['Mechanistic Error'])))
	hybrid_mech_err_means.append(statistics.mean(list(temp_hybrid['Mechanistic Error'])))
	print(statistics.mean(list(temp_node['Mechanistic Error'])))
	node_mech_err_means.append(statistics.mean(list(temp_node['Mechanistic Error'])))
	print()

plt.figure(figsize=(8,6))

plt.title('Average RMSE Comparison',fontsize=16)
plt.plot(unique_dims, node_mech_err_means,label="Pure NODE")
plt.plot(unique_dims, hybrid_mech_err_means,label="Hybrid NODE")
plt.xlabel('Network Size',fontsize=14)
plt.ylabel('Mean Mechanistic RMSE',fontsize=14)
plt.legend(fontsize=14)
plt.savefig("MeanMechanisticRMSE_Plots.png")
"""
"""
plt.figure(figsize=(8,6))
plt.title('Variance RMSE Comparison',fontsize=16)
plt.plot(unique_dims, NeuralODE_Summary['Var RMSE'],label="Neural ODE")
plt.plot(unique_dims, HybridODE_Summary['Var RMSE'],label="Hybrid ODE")
plt.xlabel('Network Size',fontsize=14)
plt.ylabel('Var RMSE',fontsize=14)
plt.legend(fontsize=14)
plt.savefig("VarRMSE_Plots.png")
"""
