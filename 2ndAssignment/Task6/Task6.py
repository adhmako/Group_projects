import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

communities = pd.read_csv("communities.data", header=None, sep=",", engine='python')

#Changing "?" missing values to be shown as NA
communities = communities.replace('?', np.nan)


#Droping NA values
communities = communities.dropna()
#print (communities)


y = communities.iloc[:, 127]
X = communities.iloc[:, [17,26,27,31,32,37,76,90,95] ]

#print(X)

communities.insert(0, 'b0', 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=1551)

X_scale = MinMaxScaler().fit(X_train)
X_train_trans = X_scale.transform(X_train) # fit on training set and transform the data
X_train = pd.DataFrame(X_train_trans, columns = list(X_train.columns)) # convert matrix to data frame with columns

y_scale = MinMaxScaler().fit(np.array(y_train).reshape(-1, 1))
y_train = y_scale.transform(np.array(y_train).reshape(-1, 1))

# Scale the test set using the X and y scalers
X_test_trans = X_scale.transform(X_test)
X_test = pd.DataFrame(X_test_trans, columns = list(X_test.columns))
y_test = y_scale.transform(np.array(y_test).reshape(-1, 1))
y_test = y_test.flatten()




X_train = np.column_stack(([1]*X_train.shape[0], X_train)) # add a column with ones for the bias value while converting it into a matrix
m,n = X_train.shape
theta = np.array([1] * n) # initial theta
X = np.array(X_train) # convert X_train into a numpy matrix
y = y_train.flatten() # convert y into an array

alpha = 0.1 # alpha value 
iteration = 100 # iterations
cost = [] # list to store cost values
theta_new = [] # list to store updates coeffient values

for i in range(0, iteration):
    pred = np.matmul(X,theta) # Calculate predicted value
    J = 1/2 * ((np.square(pred - y)).mean()) # Calculate cost function
   
    t_cols = 0 # iteration for theta values
    
    # Update the theta values for all the features with the gradient of the cost function
    for t_cols in range(0,n): 
        t = round(theta[t_cols] - alpha/m * sum((pred-y)*X[:,t_cols]),4) # calculate new theta value
        theta_new.append(t) # save new theta values in a temporary array
        
# update theta array
    theta = [] # empty the theta array
    theta = theta_new # assign new values of theta to array
    theta_new = [] # empty temporary array
    cost.append(J) # append cost function to the cost array

plt.figure(figsize=(10,8))
plt.plot(cost)
plt.title('Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value')
#plt.savefig('plot.pdf')
None
print(theta)
plt.show()


