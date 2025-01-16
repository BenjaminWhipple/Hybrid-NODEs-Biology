library(dplyr)
library(broom)
library(xtable)

### LOAD DATA
Glycolysis_known_param_hybrid_data = read.csv("../Mechanism_Uncertainty/Glycolysis_Analysis/HyperparameterExperiments/Glycolysis_KnownParamHybrid_Results.csv")
Glycolysis_unknown_param_hybrid_data = read.csv("../Mechanism_Uncertainty/Glycolysis_Analysis/HyperparameterExperiments/Glycolysis_UnknownParamHybrid_Results.csv")
Glycolysis_NODE_data = read.csv("../Mechanism_Uncertainty/Glycolysis_Analysis/HyperparameterExperiments/Glycolysis_NODE_Results.csv")

LV_3Species_known_param_hybrid_data = read.csv("../State_Uncertainty/3_Species_LV/HyperparameterExperiments/3Species_LV_KnownParamHybrid_Results.csv")
LV_3Species_unknown_param_hybrid_data = read.csv("../State_Uncertainty/3_Species_LV/HyperparameterExperiments/3Species_LV_UnknownParamHybrid_Results.csv")
LV_3Species_NODE_data = read.csv("../State_Uncertainty/3_Species_LV/HyperparameterExperiments/3Species_LV_NODE_Results.csv")

### Process Data
# Add a column indicating which experiment, and add model structure identifier for each experiment

Glycolysis_known_param_hybrid_data$Experiment = rep("Glycolysis",nrow(Glycolysis_known_param_hybrid_data))
Glycolysis_unknown_param_hybrid_data$Experiment = rep("Glycolysis",nrow(Glycolysis_unknown_param_hybrid_data))
Glycolysis_NODE_data$Experiment = rep("Glycolysis",nrow(Glycolysis_NODE_data))
LV_3Species_known_param_hybrid_data$Experiment = rep("LV_3Species",nrow(LV_3Species_known_param_hybrid_data))
LV_3Species_unknown_param_hybrid_data$Experiment = rep("LV_3Species",nrow(LV_3Species_unknown_param_hybrid_data))
LV_3Species_NODE_data$Experiment = rep("LV_3Species",nrow(LV_3Species_NODE_data))

Glycolysis_known_param_hybrid_data$Structure = rep("KnownHybrid",nrow(Glycolysis_known_param_hybrid_data))
Glycolysis_unknown_param_hybrid_data$Structure = rep("UnknownHybrid",nrow(Glycolysis_unknown_param_hybrid_data))
Glycolysis_NODE_data$Structure = rep("NODE",nrow(Glycolysis_NODE_data))
LV_3Species_known_param_hybrid_data$Structure = rep("KnownHybrid",nrow(LV_3Species_known_param_hybrid_data))
LV_3Species_unknown_param_hybrid_data$Structure = rep("UnknownHybrid",nrow(LV_3Species_unknown_param_hybrid_data))
LV_3Species_NODE_data$Structure = rep("NODE",nrow(LV_3Species_NODE_data))

# Merge Data-sets
HyperParameterExperimentData = rbind(Glycolysis_known_param_hybrid_data,
                                     Glycolysis_unknown_param_hybrid_data,
                                     Glycolysis_NODE_data,
                                     LV_3Species_known_param_hybrid_data,
                                     LV_3Species_unknown_param_hybrid_data,
                                     LV_3Species_NODE_data)

# Remove "inf" values (only removes 8 observations.)
HyperParameterExperimentData <- HyperParameterExperimentData[!is.infinite(HyperParameterExperimentData$Train.Loss),]

write.csv(HyperParameterExperimentData,"HyperparameterExperimentsData.csv")


# Create factors of variables
HyperParameterExperimentData["Size_Factor"]=factor(HyperParameterExperimentData[["Size"]],levels=c("5","15","25"),ordered=T)
HyperParameterExperimentData["Batch_Time_Factor"]=factor(HyperParameterExperimentData[["Batch.Time"]],ordered=T)
HyperParameterExperimentData["Batch_Size_Factor"]=factor(HyperParameterExperimentData[["Batch.Size"]],ordered=T)
HyperParameterExperimentData["Learning_Rate_Factor"]=factor(HyperParameterExperimentData[["Learning.Rate"]],ordered=T)
HyperParameterExperimentData["Learning_Rate_Step_Factor"]=factor(HyperParameterExperimentData[["Learning.Rate.Step"]],ordered=T)
HyperParameterExperimentData["Iterations_Factor"]=factor(HyperParameterExperimentData[["Iterations"]],ordered=T)

HyperParameterExperimentData["Experiment_Factor"]=factor(HyperParameterExperimentData[["Experiment"]])
HyperParameterExperimentData["Structure_Factor"]=factor(HyperParameterExperimentData[["Structure"]])


# (Size_Factor,Batch_Time_Factor,Batch_Size_Factor,Learning_Rate_Factor,Learning_Rate_Step_Factor,Iterations_Factor,Experiment_Factor,Structure_Factor)
clean_data <- na.omit(HyperParameterExperimentData)

my_anova <- aov(log10(Train.Loss) ~ Experiment*Structure + (Size_Factor + Batch_Time_Factor + Batch_Size_Factor + Learning_Rate_Factor + Learning_Rate_Step_Factor + Iterations_Factor)^2,data = clean_data)
summary(my_anova)

# Extract the sum of squares from the model
ss_total <- sum((clean_data$"Train.Loss" - mean(clean_data$"Train.Loss"))^2)  # Total SS
ss_residual <- sum(my_anova$residuals^2)  # Residual SS

# Calculate R2
r_squared <- 1 - (ss_residual / ss_total)
r_squared

xtable(my_anova)
