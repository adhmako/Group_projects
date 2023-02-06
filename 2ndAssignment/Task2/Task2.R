
#install.packages("readxl")
library ("readxl")

setwd("C:/Master/BD/Projects/Project2")

#Inserting the dataset
dta <- read.table("communities.data", fileEncoding="UTF-8", sep=",", header=FALSE)

#Renaming the columns of the dataset
d1 <- read_excel("names.xlsx")
colnames(dta) <- d1$names

#Changing "?" missing values to be shown as NA
idx <- dta == "?"
is.na(dta) <- idx

#Droping NA values
data <- na.omit(dta)

#Exporting the dataset to use it as a csv file in python as well
#write.csv(data,"C:/Master/BD/Projects/Project2/data.csv", row.names = FALSE)


str(data)
summary(data)

#OLS
model <- lm(ViolentCrimesPerPop ~ medIncome + whitePerCap + blackPerCap + HispPerCap + NumUnderPov + PctUnemployed + HousVacant + MedRent + NumStreet, data= data)
summary (model)
print (model)
