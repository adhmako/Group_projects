# calculateSSR
# predictedValues: predicted values for the dependent variable
# actualValues: actual values of the dependent vairable

calculateSSR<-function(predictedValues, actualValues){
  err<- sum( (actualValues - predictedValues)^2  ) 
  return( err )
}

# createRandomDataFrame
# Parameters
# numVariables: number of variables of the dataframe
# numObservations: observations of the dataset
# minValue: the minimum value an observation can take
# maxValue: the maximum value an observation can take

createRandomDataFrame<-function( numVariables=15, numObservations=80, minValue=10, maxValue=50){
  
  data <- replicate(numVariables, runif(numObservations, min=minValue, max=maxValue))
  return( as.data.frame(data) )
}


RSquared = data.frame(nVars=numeric(0), nObs=numeric(0), RSquared=numeric(0))


#nVars: number of variables
for (nVars in 2:15) {
  # number of observations
  for (nObs in (nVars+1):80) {
    #
    # running an ols
    #
    cat( sprintf("Running OLS with: %d variables and %d observations...", nVars, nObs)  )

    if (nObs*0.3 < 1){
      print('SKIPPING')
      next
    }  
    
    if ( (nObs - nObs*0.3) < nVars){
      print('SKIPPING')
      next
    }
    
    df <- createRandomDataFrame(nVars, nObs)
    
    testDataPoints <- sample(nrow(df), nObs*0.3)
    trainDF<-df[-testDataPoints,]
    testDF<-df[testDataPoints,]
    
    estimation <- lm(V1 ~ ., data=trainDF)
    cat( sprintf("R-squared: %f\n", summary(estimation)$r.squared)  )
    RSquared[ nrow(RSquared)+1, ] <- c( nVars, nObs, summary(estimation)$r.squared)
  }
  
}

cat( sprintf("Total of %d", nrow(RSquared[ which(RSquared[,"RSquared"] >=0.80), ])  ) )

print( RSquared[ which(RSquared[,"RSquared"] >=0.80), ] )

# One of the cases where overfitting occurs is when we have 11 variables and 18 observations, since the R squared
# of the OLS estimation is 0.9 (> 0.8)
# We will perform an ols for this dataset in order to see the overfitting via the generalization and training error.


df <- createRandomDataFrame(11, 18, 10, 50)
testDataPoints <- sample(nrow(df), 18*0.3)
trainDF<-df[-testDataPoints,]
testDF<-df[testDataPoints,]

#Ols
estimation <- lm(V1 ~ ., data=trainDF)
print( sprintf("R-Squared: %f", summary(estimation)$r.squared) )

# training error
predictedTrain<-predict(estimation, trainDF)
trainingError<-calculateSSR(predictedTrain, trainDF$V1)
print( sprintf("Train error=%.3f", trainingError))


# gerenalization error
predictedTest<-predict(estimation, testDF)
generalizationError<-calculateSSR(predictedTest, testDF$V1)
print( sprintf("Generalization error=%.3f", generalizationError))

# class difference: generalization error/ training error
class = generalizationError/trainingError
print (class)

