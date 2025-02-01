import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

glycolysis_params = ["J0","k2","k3","k4","k5","k6","k","kappa","psi","N","A"]
threespecies_params = ["beta","gamma","delta"]

def lower_half_median(x):
	x_sorted = x.sort_values()
	lower_half = x_sorted.iloc[:len(x_sorted)//2]
	return lower_half.median()

def percentiles(x):
	return pd.Series({
		"Q1":x.quantile(0.05),
		"Q2":x.quantile(0.5),
		"Q3":x.quantile(0.95)
		})

def bootstrap_median(x,n_bootstrap = 1000):
	medians = [x.sample(frac=1,replace=True).median() for _ in range(n_bootstrap)]
	return pd.Series({
		"Median":x.median(),
		"Lower":np.percentile(medians,2.5),
		#"Middle":np.percentile(medians,50.0),
		"Upper":np.percentile(medians,97.5)
		})

"""
Decision Rule: 
- X dominates Y if the quartiles of X are lower than the quartiles of Y. 
- If we are left with a set in which no dominance exists, lexicographically order the set for smaller width, smaller batch time, and larger learning rate. This ordering corresponds to computationally simpler models.
"""
def Find_Dominant_Parameter_Set(input_df):
	# Find dfs for which lower and upper bounds 
	dominant_set_CIs = []
	for a in input_df.index:
		for b in input_df.index:
			if a!=b and input_df.loc[a,"Upper"] < input_df.loc[b,"Lower"]:
				dominant_set_CIs.append(a)

	# Consider the case where everything is roughly equivalent.
	if len(dominant_set_CIs) == 0:
		dominant_set_CIs = [a for a in input_df.index]

	# Simple way to remove duplicates
	dominant_set_CIs = list(set(dominant_set_CIs))

	# Ensure no existing element in the dominant set dominates a given element.
	survives = []
	for y in dominant_set_CIs:
		if len([x for x in dominant_set_CIs if input_df.loc[x,"Upper"] < input_df.loc[y,"Lower"]])==0:
			survives.append(y)

	# Consider the case where everything is roughly equivalent.
	if len(survives)==0:
		survives = dominant_set_CIs

	# Select for lexicographic ordering. This is a strict order. Eventually, this code should be generalized.
	#print(survives)
	#print(dominant_set_CIs)
	best = survives[0]
	for i in survives:
		if i[0] < best[0]:
			best = i
		elif ((i[0] == best[0]) and (i[1] < best[1])):
			best = i
		elif ((i[0] == best[0]) and (i[1] == best[1]) and (i[2] < best[2])):
			best = i
		else:
			continue

	return best, survives

Glycolysis_Performances = pd.read_csv("SerializedObjects/DataFrames/Glycolysis_Results_Summary.csv").dropna()
ThreeSpecies_Performances = pd.read_csv("SerializedObjects/DataFrames/ThreeSpecies_Results_Summary.csv").dropna()

## Glycolysis
glycolysis_unknown_param_rmse = Glycolysis_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["unknown_param_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
glycolysis_unknown_param_rmse_best_params = Find_Dominant_Parameter_Set(glycolysis_unknown_param_rmse)
glycolysis_unknown_param_rmse_typical = bootstrap_median(Glycolysis_Performances["unknown_param_test_rmse"])
glycolysis_unknown_param_rmse_best = glycolysis_unknown_param_rmse.loc[glycolysis_unknown_param_rmse_best_params[0]]
print(glycolysis_unknown_param_rmse_best)

glycolysis_strict_known_param_rmse = Glycolysis_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["known_param_strict_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
glycolysis_strict_known_param_rmse_best_params = Find_Dominant_Parameter_Set(glycolysis_strict_known_param_rmse)
glycolysis_strict_known_param_rmse_typical = bootstrap_median(Glycolysis_Performances["known_param_strict_test_rmse"])
glycolysis_strict_known_param_rmse_best = glycolysis_strict_known_param_rmse.loc[glycolysis_strict_known_param_rmse_best_params[0]]
print(glycolysis_strict_known_param_rmse_best)

glycolysis_loose_known_param_rmse = Glycolysis_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["known_param_loose_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
glycolysis_loose_known_param_rmse_best_params = Find_Dominant_Parameter_Set(glycolysis_loose_known_param_rmse)
glycolysis_loose_known_param_rmse_typical = bootstrap_median(Glycolysis_Performances["known_param_loose_test_rmse"])
glycolysis_loose_known_param_rmse_best = glycolysis_loose_known_param_rmse.loc[glycolysis_loose_known_param_rmse_best_params[0]]
print(glycolysis_loose_known_param_rmse_best)

glycolysis_NODE_test_rmse = Glycolysis_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["NODE_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
glycolysis_NODE_rmse_best_params = Find_Dominant_Parameter_Set(glycolysis_NODE_test_rmse)
glycolysis_NODE_rmse_typical = bootstrap_median(Glycolysis_Performances["NODE_test_rmse"])
glycolysis_NODE_rmse_best = glycolysis_NODE_test_rmse.loc[glycolysis_NODE_rmse_best_params[0]]
print(glycolysis_NODE_rmse_best)

## Three Species
threespecies_unknown_param_rmse = ThreeSpecies_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["unknown_param_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
threespecies_unknown_param_rmse_best_params = Find_Dominant_Parameter_Set(threespecies_unknown_param_rmse)
threespecies_unknown_param_rmse_typical = bootstrap_median(ThreeSpecies_Performances["unknown_param_test_rmse"])
threespecies_unknown_param_rmse_best = threespecies_unknown_param_rmse.loc[threespecies_unknown_param_rmse_best_params[0]]
print(threespecies_unknown_param_rmse_best)

threespecies_strict_known_param_rmse = ThreeSpecies_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["known_param_strict_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
threespecies_strict_known_param_rmse_best_params = Find_Dominant_Parameter_Set(threespecies_strict_known_param_rmse)
threespecies_strict_known_param_rmse_typical = bootstrap_median(ThreeSpecies_Performances["known_param_strict_test_rmse"])
threespecies_strict_known_param_rmse_best = threespecies_strict_known_param_rmse.loc[threespecies_strict_known_param_rmse_best_params[0]]
print(threespecies_strict_known_param_rmse_best)

threespecies_loose_known_param_rmse = ThreeSpecies_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["known_param_loose_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
threespecies_loose_known_param_rmse_best_params = Find_Dominant_Parameter_Set(threespecies_loose_known_param_rmse)
threespecies_loose_known_param_rmse_typical = bootstrap_median(ThreeSpecies_Performances["known_param_loose_test_rmse"])
threespecies_loose_known_param_rmse_best = threespecies_loose_known_param_rmse.loc[threespecies_loose_known_param_rmse_best_params[0]]
print(threespecies_loose_known_param_rmse_best)

threespecies_NODE_test_rmse = ThreeSpecies_Performances.groupby(by=['width', 'batch-time', 'learning-rate'])["NODE_test_rmse"].apply(bootstrap_median).unstack(level=-1).rename(columns={"level_3":"Percentiles"})
threespecies_NODE_rmse_best_params = Find_Dominant_Parameter_Set(threespecies_NODE_test_rmse)
threespecies_NODE_rmse_typical = bootstrap_median(ThreeSpecies_Performances["NODE_test_rmse"])
threespecies_NODE_rmse_best = threespecies_NODE_test_rmse.loc[threespecies_NODE_rmse_best_params[0]]
print(threespecies_NODE_rmse_best)

models = ["Unknown", "Strict","Loose","NDE"]

glycoylsis_values = [list(glycolysis_unknown_param_rmse_best),list(glycolysis_strict_known_param_rmse_best),list(glycolysis_loose_known_param_rmse_best),list(glycolysis_NODE_rmse_best)]
threespecies_values = [list(threespecies_unknown_param_rmse_best),list(threespecies_strict_known_param_rmse_best),list(threespecies_loose_known_param_rmse_best),list(threespecies_NODE_rmse_best)]


# Create figure and axis
fig, ax = plt.subplots(1,2,sharey=True)

y_positions = range(len(models))

ax[0].set_title("Glycolysis")
ax[0].hlines(y=y_positions, xmin=[i[1] for i in glycoylsis_values], xmax=[i[2] for i in glycoylsis_values], color='royalblue')
ax[0].scatter([i[0] for i in glycoylsis_values], y_positions, color='royalblue', zorder=3)  # Add median points
ax[0].set_xscale("log")
ax[0].set_yticks(y_positions)
ax[0].set_yticklabels(models)
ax[0].set_xlabel('RMSE on Test Set')
ax[0].set_xticks([1e-2,1e-1,1e-0])

ax[1].set_title("Three Species Lotka Volterra")
ax[1].hlines(y=y_positions, xmin=[i[1] for i in threespecies_values], xmax=[i[2] for i in threespecies_values], color='royalblue')
ax[1].scatter([i[0] for i in threespecies_values], y_positions, color='royalblue', zorder=3)  # Add median points
ax[1].set_xlabel('RMSE on Test Set')
ax[1].set_xscale("log")
ax[1].set_xticks([1e-2,1e-1,1e-0])
plt.savefig("Model_Performances.png")

## Create similar plots for parameter errors.
Glycolysis_Unknown_Parameter_Estimates = pd.read_csv("SerializedObjects/DataFrames/Glycolysis_Unknown_Parameter_Estimates.csv").dropna()
ThreeSpecies_Unknown_Parameter_Estimates = pd.read_csv("SerializedObjects/DataFrames/ThreeSpecies_Unknown_Parameter_Estimates.csv").dropna()

glycolysis_best_parameter_fits = Glycolysis_Unknown_Parameter_Estimates[(Glycolysis_Unknown_Parameter_Estimates[["width","batch-time","learning-rate"]]==glycolysis_unknown_param_rmse_best_params[0]).all(axis=1)].dropna()
threespecies_best_parameter_fits = ThreeSpecies_Unknown_Parameter_Estimates[(ThreeSpecies_Unknown_Parameter_Estimates[["width","batch-time","learning-rate"]]==threespecies_unknown_param_rmse_best_params[0]).all(axis=1)].dropna()

# Glycolysis
fig, ax = plt.subplots()
y_positions = range(len(glycolysis_params))
percentile_list = []
for param in glycolysis_params:
	percentile_list.append(list(percentiles(glycolysis_best_parameter_fits[param+"_percent_err"])))

ax.set_title("Glycolysis Parameter Estimate Accuracy")
ax.hlines(y=y_positions,xmin=[i[0] for i in percentile_list],xmax=[i[2] for i in percentile_list],color="royalblue")
ax.scatter([i[1] for i in percentile_list],y_positions,color="royalblue",zorder=3)
ax.set_xscale("log")
ax.set_yticks(y_positions)
ax.set_yticklabels(glycolysis_params)
ax.set_xlabel("Percent Error")
plt.savefig("Glycolysis_ParameterEstimate_Accuracy.png")

# Three Species Lotka-Volterra
fig, ax = plt.subplots()
y_positions = range(len(threespecies_params))
percentile_list = []
for param in threespecies_params:
	percentile_list.append(list(percentiles(threespecies_best_parameter_fits[param+"_percent_err"])))

ax.set_title("Three Species Lotka-Volterra Parameter Estimate Accuracy")
ax.hlines(y=y_positions,xmin=[i[0] for i in percentile_list],xmax=[i[2] for i in percentile_list],color="royalblue")
ax.scatter([i[1] for i in percentile_list],y_positions,color="royalblue",zorder=3)
ax.set_xscale("log")
ax.set_yticks(y_positions)
ax.set_yticklabels(threespecies_params)
ax.set_xlabel("Percent Error")
plt.savefig("ThreeSpecies_ParameterEstimate_Accuracy.png")