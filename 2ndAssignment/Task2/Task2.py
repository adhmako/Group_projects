#subtask 2i
import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt

#Inserting the dataset
dta = pd.read_csv("communities.data", header=None, sep=",")
print (dta.dtypes)
print (dta)

d1 = pd.read_excel("names.xlsx")
print (d1)

#Renaming the columns of the dataset
dta.columns = d1['names']
print (dta)

#Changing "?" missing values to be shown as NA
dta = dta.replace('?', np.nan)
print (dta)

#Droping NA values
data = dta.dropna()
print (data.dtypes)
print (data) 


#OLS
independentVariables = data.loc[:, ['medIncome', 'whitePerCap', 'blackPerCap', 'HispPerCap', 'NumUnderPov', 'PctUnemployed', 'HousVacant', 'MedRent', 'NumStreet']]
dependentVariable = data.loc[:, 'ViolentCrimesPerPop']


lm = LinearRegression(normalize=False, fit_intercept=True)
model = lm.fit(independentVariables, dependentVariable)

print("Coefficients:", model.coef_)
# print("Note: Coefficients should be interpreted, based on the order of variables in the data.frame of independent variables. This means that:")
# print("\t-Coefficient b1 (medIncome):", model.coef_[0])
# print("\t-Coefficient b2 (whitePerCap):", model.coef_[1])

print("\t-Intercept:", model.intercept_)

# print("You may also display the R-squared (the proportion of variance explained):")

#Calculate R-squared
Rsquared = lm.score(independentVariables, dependentVariable)

print("R-squared:", Rsquared)

from regressors import stats

print("\n====== Summary statistics ======\n")
stats.summary(model, independentVariables, dependentVariable)

