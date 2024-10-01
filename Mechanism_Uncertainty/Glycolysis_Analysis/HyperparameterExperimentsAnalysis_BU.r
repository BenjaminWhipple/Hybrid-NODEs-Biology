library(dplyr)
library(broom)
library(xtable)
### LOAD DATA
known_param_hybrid_data = read.csv("HyperparameterExperiments/Glycolysis_KnownParamHybrid_Results.csv")
unknown_param_hybrid_data = read.csv("HyperparameterExperiments/Glycolysis_UnknownParamHybrid_Results.csv")
NODE_data = read.csv("HyperparameterExperiments/Glycolysis_NODE_Results.csv")

### COUNT MISSING
print(sum(is.na(known_param_hybrid_data["Train.Loss"])))
print(sum(is.na(unknown_param_hybrid_data["Train.Loss"])))
print(sum(is.na(NODE_data["Train.Loss"])))

### COUNT FULLY MISSING STRATUM
known_param_res <- known_param_hybrid_data %>%
  group_by("Size","Batch.Time","Batch.Size","Learning.Rate","Learning.Rate.Step","Iterations") %>%
  summarize(all_missing=all(is.na("Train.Loss"))) %>%
  filter(all_missing)

unknown_param_res <- unknown_param_hybrid_data %>%
  group_by("Size","Batch.Time","Batch.Size","Learning.Rate","Learning.Rate.Step","Iterations") %>%
  summarize(all_missing=all(is.na("Train.Loss"))) %>%
  filter(all_missing)

NODE_param_res <- NODE_data %>%
  group_by("Size","Batch.Time","Batch.Size","Learning.Rate","Learning.Rate.Step","Iterations") %>%
  summarize(all_missing=all(is.na("Train.Loss"))) %>%
  filter(all_missing)

known_param_res
unknown_param_res
NODE_param_res

# So, we can see that there is no compartment that is fully missing entries.

### CREATE MODEL OF PARAMETER INFLUENCE ON FIT OUTCOME (TRAINING LOSS)

# Create missing values columns

unknown_param_hybrid_data["Missing"] = ifelse(is.na(unknown_param_hybrid_data["Train.Loss"]),1,0)
known_param_hybrid_data["Missing"] = ifelse(is.na(known_param_hybrid_data["Train.Loss"]),1,0)
NODE_data["Missing"] = ifelse(is.na(NODE_data["Train.Loss"]),1,0)

# Encode factor levels

cols = c("Size","Batch.Time","Batch.Size","Learning.Rate","Learning.Rate.Step","Iterations")
factor_cols = c("Size_Factor","Batch_Time_Factor","Batch_Size_Factor","Learning_Rate_Factor","Learning_Rate_Step_Factor","Iterations_Factor")
print(cols)

unknown_param_hybrid_data["Size_Factor"]=factor(unknown_param_hybrid_data[["Size"]],levels=c("5","15","25"),ordered=T)
unknown_param_hybrid_data["Batch_Time_Factor"]=factor(unknown_param_hybrid_data[["Batch.Time"]],ordered=T)
unknown_param_hybrid_data["Batch_Size_Factor"]=factor(unknown_param_hybrid_data[["Batch.Size"]],ordered=T)
unknown_param_hybrid_data["Learning_Rate_Factor"]=factor(unknown_param_hybrid_data[["Learning.Rate"]],ordered=T)
unknown_param_hybrid_data["Learning_Rate_Step_Factor"]=factor(unknown_param_hybrid_data[["Learning.Rate.Step"]],ordered=T)
unknown_param_hybrid_data["Iterations_Factor"]=factor(unknown_param_hybrid_data[["Iterations"]],ordered=T)

known_param_hybrid_data["Size_Factor"]=factor(known_param_hybrid_data[["Size"]],levels=c("5","15","25"),ordered=T)
known_param_hybrid_data["Batch_Time_Factor"]=factor(known_param_hybrid_data[["Batch.Time"]],ordered=T)
known_param_hybrid_data["Batch_Size_Factor"]=factor(known_param_hybrid_data[["Batch.Size"]],ordered=T)
known_param_hybrid_data["Learning_Rate_Factor"]=factor(known_param_hybrid_data[["Learning.Rate"]],ordered=T)
known_param_hybrid_data["Learning_Rate_Step_Factor"]=factor(known_param_hybrid_data[["Learning.Rate.Step"]],ordered=T)
known_param_hybrid_data["Iterations_Factor"]=factor(known_param_hybrid_data[["Iterations"]],ordered=T)

NODE_data["Size_Factor"]=factor(NODE_data[["Size"]],levels=c("5","15","25"),ordered=T)
NODE_data["Batch_Time_Factor"]=factor(NODE_data[["Batch.Time"]],ordered=T)
NODE_data["Batch_Size_Factor"]=factor(NODE_data[["Batch.Size"]],ordered=T)
NODE_data["Learning_Rate_Factor"]=factor(NODE_data[["Learning.Rate"]],ordered=T)
NODE_data["Learning_Rate_Step_Factor"]=factor(NODE_data[["Learning.Rate.Step"]],ordered=T)
NODE_data["Iterations_Factor"]=factor(NODE_data[["Iterations"]],ordered=T)

# Run ANOVA

log_model <- glm(formula = Missing ~ (Size_Factor + Batch_Time_Factor + Batch_Size_Factor + Learning_Rate_Factor + Learning_Rate_Step_Factor + Iterations_Factor)^2, data = unknown_param_hybrid_data, family = "binomial")
summary(log_model)

res_anova <- aov(log10(Train.Loss) ~ (Size_Factor + Batch_Time_Factor + Batch_Size_Factor + Learning_Rate_Factor + Learning_Rate_Step_Factor + Iterations_Factor)^2,data = unknown_param_hybrid_data)
summary(res_anova)
res_anova$coefficients
nrow(tidy(log_model))
xtable(res_anova)
library(dplyr)
library(rsample)
library(broom)
library(splitstackshape)

test <- unknown_param_hybrid_data %>%
  mutate(strata = interaction(unknown_param_hybrid_data$"Size_Factor",
                              unknown_param_hybrid_data$"Batch_Time_Factor",
                              unknown_param_hybrid_data$"Batch_Size_Factor",
                              unknown_param_hybrid_data$"Learning_Rate_Factor",
                              unknown_param_hybrid_data$"Learning_Rate_Step_Factor",
                              unknown_param_hybrid_data$"Iterations_Factor",
                              sep="_"
                              )
  )

test$strata = factor(test$"strata")
test$strata



out = stratified(test,c("strata"),9,replace=TRUE)
out  
log_model <- glm(formula = log10(Missing) ~ (Size_Factor + Batch_Time_Factor + Batch_Size_Factor + Learning_Rate_Factor + Learning_Rate_Step_Factor + Iterations_Factor)^2, data = out, family = "binomial")
summary(log_model)
test3 = tidy(log_model)
test2 = matrix(0,nrow=nrow(tidy(log_model)),10)
test2
print(as.matrix(test3["estimate"]))
test2[,2]=as.matrix(test3["estimate"])
test2
#test2[,2]=tidy(log_model)
print(sum(is.na(as.data.frame(out)["Train.Loss"])))
print(sum(is.na(out2["Train.Loss"])))