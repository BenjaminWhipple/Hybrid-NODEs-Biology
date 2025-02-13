"""
This code merges the json from the individual experiments into a dataframes for further analysis.
"""

import os
import json
import pandas as pd
import numpy as np

glycolysis_directory = "Glycolysis_Results"
threespecies_directory = "ThreeSpecies_LV_Results"
hyper_params = ["width","batch-time","learning-rate","reg-param","parameter-index","sample","replicate"] # Appear in file-names

glycolysis_filenames = os.listdir(glycolysis_directory)
threespecies_filenames = os.listdir(threespecies_directory)

### GLYCOLYSIS

## LOAD ALL FILES
glycolysis_data = []
for filename in glycolysis_filenames:
	filepath = glycolysis_directory+"/"+filename
	with open(filepath, "r") as f:
		data = json.load(f)

	glycolysis_data.append((filename,data))

## FIGURE OUT KEYS TO USE
glycolysis_keys = [key for key, value in glycolysis_data[0][1].items()]
main_keys = [key for key in glycolysis_keys if "dict" not in key]
param_keys = [key for key in glycolysis_keys if "dict" in key] # We will need to do additional processing with these.
param_terms = [key for key, value in glycolysis_data[0][1][param_keys[0]].items()]

## PROCESS ALL FILES
glycolysis_main_data = []
glycolysis_unknown_param_data = []
glycolysis_full = []

columns = hyper_params + main_keys
param_columns = hyper_params + param_terms
full_columns = hyper_params + main_keys + param_terms

for data in glycolysis_data:
	# Get hyper-parameters
	filename, this_data = data
	these_hyper_params = [float(i) for i in filename.replace(".json","").split("_")[1:]]
	
	# Get data
	main_key_data = [this_data[key] for key in main_keys]
	unknown_data = [this_data["unknown_err_dict"][key] for key in param_terms]
	
	# Construct and append rows
	this_main_row = these_hyper_params + main_key_data
	this_unknown_row = these_hyper_params + unknown_data
	this_full_row = these_hyper_params + main_key_data + unknown_data
	
	glycolysis_main_data.append(this_main_row)
	glycolysis_unknown_param_data.append(this_unknown_row)
	glycolysis_full.append(this_full_row)

## CONSTRUCT AND SAVE DATAFRAMES
glycolysis_main_df = pd.DataFrame(data = glycolysis_main_data, columns=columns)
glycolysis_unknown_param_df = pd.DataFrame(data=glycolysis_unknown_param_data, columns=param_columns)
glycolysis_full_df = pd.DataFrame(data=glycolysis_full, columns=full_columns)

glycolysis_main_df.to_csv("DataFrames/Glycolysis_Results_Summary.csv", index = False)
glycolysis_unknown_param_df.to_csv("DataFrames/Glycolysis_Unknown_Parameter_Estimates.csv", index = False)
glycolysis_full_df.to_csv("DataFrames/Glycolysis_Full.csv", index = False)

### THREE SPECIES LV

## LOAD ALL FILES
threespecies_data = []
for filename in threespecies_filenames:
	filepath = threespecies_directory+"/"+filename
	with open(filepath, "r") as f:
		data = json.load(f)

	threespecies_data.append((filename,data))

## FIGURE OUT KEYS TO USE
threespecies_keys = [key for key, value in threespecies_data[0][1].items()]
main_keys = [key for key in threespecies_keys if "dict" not in key]
param_keys = [key for key in threespecies_keys if "dict" in key] # We will need to do additional processing with these.
param_terms = [key for key, value in threespecies_data[0][1][param_keys[0]].items()]

## PROCESS ALL FILES
threespecies_main_data = []
threespecies_unknown_param_data = []
threespecies_full = []

columns = hyper_params + main_keys
param_columns = hyper_params + param_terms
full_columns = hyper_params + main_keys + param_terms

for data in threespecies_data:
	# Get hyper-parameters
	filename, this_data = data
	these_hyper_params = [float(i) for i in filename.replace(".json","").split("_")[1:]]
	
	# Get data
	main_key_data = [this_data[key] for key in main_keys]
	unknown_data = [this_data["unknown_err_dict"][key] for key in param_terms]
	
	# Construct and append rows
	this_main_row = these_hyper_params + main_key_data
	this_unknown_row = these_hyper_params + unknown_data
	this_full_row = these_hyper_params + main_key_data + unknown_data

	threespecies_main_data.append(this_main_row)
	threespecies_unknown_param_data.append(this_unknown_row)
	threespecies_full.append(this_full_row)

## CONSTRUCT AND SAVE DATAFRAMES
threespecies_main_df = pd.DataFrame(data = threespecies_main_data, columns=columns)
threespecies_unknown_param_df = pd.DataFrame(data=threespecies_unknown_param_data, columns=param_columns)
threespecies_full_df = pd.DataFrame(data=threespecies_full, columns=full_columns)

threespecies_main_df.to_csv("DataFrames/ThreeSpecies_Results_Summary.csv", index = False)
threespecies_unknown_param_df.to_csv("DataFrames/ThreeSpecies_Unknown_Parameter_Estimates.csv", index = False)
threespecies_full_df.to_csv("DataFrames/ThreeSpecies_Full.csv", index = False)
