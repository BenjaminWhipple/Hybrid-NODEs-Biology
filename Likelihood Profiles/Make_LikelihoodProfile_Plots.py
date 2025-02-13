import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
"""
TODO: Make plots dropping NA values first. Lets see what happens
"""

glycolysis_df = pd.read_csv("SerializedObjects/DataFrames/Glycolysis_Full.csv").dropna()
threespecies_df = pd.read_csv("SerializedObjects/DataFrames/ThreeSpecies_Full.csv").dropna()
print(glycolysis_df)
print(threespecies_df)

glycolysis_params = ["J0","k2","k3","k4","k5","k6","k","kappa","psi","N","A"]
glycolysis_params_names = ["J_0","k_2","k_3","k_4","k_5","k_6","k","\kappa","\psi","N","A"]

# J0,k1,k2,k3,k4,k5,k6,k,kappa,q,K1,psi,N,A
#glycolysis_p = np.array([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.])
glycolysis_p = np.array([2.5, 6., 16., 100., 1.28, 12., 1.8, 13., 0.1, 1., 4.])

threespecies_params = ["beta","gamma","delta"]
threespecies_params_names = [r"\beta",r"\gamma",r"\delta"]
threespecies_p = np.array([1.5,3.0,1.0])

#reg_params = list(glycolysis_df["reg-param"].unique())
#print(reg_params)
small_reg_params = [0.0,1.0]
colors = ["darkorange","firebrick"]

for i in range(len(glycolysis_params)):
	print(glycolysis_params[i])
	subdf = glycolysis_df[glycolysis_df["parameter-index"] == i]

	plt.figure()

	for j in range(len(small_reg_params)):
		param = small_reg_params[j]

		subsubdf = subdf[subdf["reg-param"]==param]

		ys = np.log10(subsubdf["unknown_param_train_rmse"]).to_numpy()
		#xs = np.log10(subsubdf[f"{glycolysis_params[i]}_fit"]).to_numpy()
		xs = subsubdf[f"{glycolysis_params[i]}_fit"].to_numpy()

		kernel = Matern(length_scale_bounds=(1e0, 5e1),nu=1.5) + WhiteKernel()
		gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
		print(xs.reshape(-1,1))
		print(ys.reshape(-1,1))
		gaussian_process.fit(xs.reshape(-1,1), ys.reshape(-1,1))
		predict_points = np.linspace(min(xs),max(xs),100)
		mean_prediction, std_prediction = gaussian_process.predict(predict_points.reshape(-1,1), return_std=True)

		plt.title(f"Glycolysis HNDE: ${glycolysis_params_names[i]}$ Profile Likelihood by $L_1$ Regularization")
		plt.ylabel("$\log_{10}$ RMSE")
		plt.xlabel(f"${glycolysis_params_names[i]}$")
		plt.plot(predict_points,mean_prediction,color=colors[j],label=f"$ \lambda_1 = {param}$ ")
		#plt.plot(xs,ys,linestyle="None",marker="o",color=colors[j],alpha=0.5)
		#plt.yscale("log")

	plt.axvline(x=glycolysis_p[i],label=f"True ${glycolysis_params_names[i]}$",color="salmon",linestyle="--")
	plt.legend()

	plt.savefig(f"Images/glycolysis_{glycolysis_params[i]}_likelihood_profile.png")

for i in range(len(threespecies_params)):
	print(threespecies_params[i])
	subdf = threespecies_df[threespecies_df["parameter-index"] == i]

	plt.figure()

	for j in range(len(small_reg_params)):
		param = small_reg_params[j]

		subsubdf = subdf[subdf["reg-param"]==param]

		ys = np.log10(subsubdf["unknown_param_train_rmse"]).to_numpy()
		#xs = np.log10(subsubdf[f"{glycolysis_params[i]}_fit"]).to_numpy()
		xs = subsubdf[f"{threespecies_params[i]}_fit"].to_numpy()

		kernel = Matern(length_scale_bounds=(1e0, 5e1),nu=1.5) + WhiteKernel()
		gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20,normalize_y = True)
		print(xs.reshape(-1,1))
		print(ys.reshape(-1,1))
		gaussian_process.fit(xs.reshape(-1,1), ys.reshape(-1,1))
		predict_points = np.linspace(min(xs),max(xs),100)
		mean_prediction, std_prediction = gaussian_process.predict(predict_points.reshape(-1,1), return_std=True)

		plt.title(f"Three Species LV HNDE: ${threespecies_params_names[i]}$ Profile Likelihood by $L_1$ Regularization")
		plt.ylabel("$\log_{10}$ RMSE")
		plt.xlabel(f"${threespecies_params_names[i]}$")
		plt.plot(predict_points,mean_prediction,color=colors[j],label=f"$ \lambda_1 = {param}$ ",)
		#plt.plot(xs,ys,linestyle="None",marker="o",color=colors[j],alpha=0.5)
		#plt.yscale("log")

	plt.axvline(x=threespecies_p[i],label=f"True ${threespecies_params_names[i]}$",color="salmon",linestyle="--")
	plt.legend()

	print(threespecies_params[i])
	plt.savefig(f"Images/threespecies_{threespecies_params[i]}_likelihood_profile.png")

