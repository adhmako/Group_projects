import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import warnings
warnings.filterwarnings('ignore')


#
# Multiply two matrices i.e. mat1 * mat2
#
def matmultiply(mat1,mat2):
    
    return( np.matmul(mat1, mat2) )
    

#
# Calculate current value of cost function J(θ).
# indV: matrix of independent variables, first column must be all 1s
# depV: matrix (dimensions nx1)of dependent variable i.e.
#
def calculateCost(indV, depV, thetas):
    return( np.sum( ((matmultiply(indV, thetas) - depV)**2) / (2*indV.shape[0]) ) )  
    

#
# Batch gradient descent
#
# indV:matrix of independent variables, first column must be all 1s
# depV: matrix (dimensions nx1)of dependent variable i.e.
# alpha: value of learning hyperparameter. Default (i.e. if no argument provided)  0.01
# numIters: number of iterations. Default (i.e. if no argument provided) 100
#
def batchGradientDescent(indV, depV, thetas, alpha = 0.01, numIters = 200, verbose = False):

     calcThetas = thetas
     
     # we store here the calculated values of J(θ)
     costHistory = pd.DataFrame( columns=["iter", "cost"])
     m = len(depV)
     #print (costHistory)
     
     for i in range(0, numIters):
       prediction=matmultiply(indV,calcThetas)
       calcThetas=calcThetas-(1/m)*alpha*(matmultiply(indV.T,(prediction-depV)))
       print(">>>> Iteration", i, ")")  
       print("       Calculate thetas...", calcThetas)
       c = calculateCost(indV, depV, calcThetas)
       print("       Calculate cost fuction for new thetas...", c)
       costHistory = costHistory.append({"iter": i, "cost": c}, ignore_index=True )

       
     # Done. Return values     
     return calcThetas, costHistory



# Read the data
communities = pd.read_csv("communities.data", header=None, sep=",", engine='python')



#That's our dependent variable
dependentVar = communities.iloc[:, 127]

# These are all our independent ones: 17,26,27,31,32,37,76,90,95
communities = communities.iloc[:, [17,26,27,31,32,37,76,90,95] ]

# Check to see if missing values are present.
communities = communities.replace('?', np.nan)
communities = communities.dropna()


# Add new column at the beginning representing the constant term b0
communities.insert(0, 'b0', 1)

# Add to a new variable to make the role of the data clearer
independentVars = communities


# Initialize thetas with some random values.
# We'll need (independentVars.shape[1])  theta values, one for each independent variable.
iniThetas = []
for i in range(0, independentVars.shape[1]):
    iniThetas.append( np.random.rand() )

initialThetas = np.array(iniThetas)

# Run BATCH gradient descent and return 2 values: I) the vector of the estimated coefficients (estimatedCoefficients) and II) the values of the
# cost function (costHistory)
estimatedCoefficients, costHistory = batchGradientDescent(independentVars.to_numpy(), dependentVar.to_numpy(), initialThetas, 0.1)

# Display now the cost function to see if alpha and number of iterations were appropriate.
costHistory.plot.scatter(x="iter", y="cost", color='red')
plt.show()

