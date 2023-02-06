# install.packages("arules")
library(arules)
library(openxlsx)

# Setup our environment. Where is our data?
setwd("C:/Master/BD/Projects/Project3")

graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

# reading the files
data <- read.xlsx("fertility-diagnosis.xlsx")

# Discretize the data set
data <- discretizeDF(data)

# Remove the 2nd and 9th columns
data <- data[,-c(2,9)]

# Create the transaction data set
transactions <- as(data, "transactions")



# Subtask i
# Run the Apriori algorithm
rules <- apriori(transactions, parameter = list(target = "rules"))

# number of rules
nrules <- length(rules)

# print the number of rules
print(nrules)

# print the rules
inspect(rules)


# Subtask ii
# Apply the Apriori algorithm
altrules <- apriori(transactions, parameter = list(supp = 0.02, conf = 1, target = "rules"),
                    appearance = list(rhs = "Diagnosis=O"))


# View the number and content of the returned rules
naltrules <- length(altrules)
print(naltrules)
print(inspect(altrules))

# For checking that alturles consists of support >= 0.02) | Optional
# altrule <- quality(altrules)
# altrule <- altrule %>% filter(support >= 0.02)

al = data.frame(
  lhs = labels(lhs(altrules)),
  rhs = labels(rhs(altrules)), 
  altrules@quality)

print(al)

# Subtask iii
# Remove redundant
finalrules <- altrules[!is.redundant(altrules)]

# Inspect
arules::inspect(finalrules)

# Create a dataframe
finalrules = data.frame(
  lhs = labels(lhs(finalrules)),
  rhs = labels(rhs(finalrules)), 
  finalrules@quality)
