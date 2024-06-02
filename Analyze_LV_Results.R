#library(walrus)

Hybrid_Results = read.csv("LV_Analysis/Results.csv")
NODE_Results = read.csv("LV_Analysis/Results_neuralODE.csv")

# Load the stringr package
library(stringr)

# Sample data with a text column containing entries like '(num1 num2)'
data <- data.frame(text_column = c("(1 2)", "(3 4)", "(5 6)"))

# Extract num1 and num2 using regular expressions
matches <- str_match(data$text_column, "\\((\\d+) (\\d+)\\)")
data$Num1 <- as.numeric(matches[, 2])
data$Num2 <- as.numeric(matches[, 3])

# View the resulting data frame
print(data)

matches <- str_match(Hybrid_Results$Layer.Dimensions,"\\((\\d+) (\\d+)\\)")
Hybrid_Results$Layer1_Dim <- as.numeric(matches[,2])
Hybrid_Results$Layer2_Dim <- as.numeric(matches[,3])


matches <- str_match(NODE_Results$Layer.Dimensions,"\\((\\d+) (\\d+)\\)")
NODE_Results$Layer1_Dim <- as.numeric(matches[,2])
NODE_Results$Layer2_Dim <- as.numeric(matches[,3])

#Need to create layer 1 and layer 2 variables.
interaction <- aov(RMSE ~ (niters + data.size + batch.time + batch.size + Layer1_Dim + Layer2_Dim)^4, data = Hybrid_Results)
summary(interaction)
interaction$coefficients

#Create a pareto plot using the coefficients.

#Hybrid_Model_Reg <- lm(RMSE ~ (niters + data.size + Layer1_Dim + Layer2_Dim)^4, Hybrid_Results)
#summary(Hybrid_Model_Reg)
