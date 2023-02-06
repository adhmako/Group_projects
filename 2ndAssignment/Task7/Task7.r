
forestfiresData <- read.csv("C:\\Users\\30697\\Desktop\\ergtzagkara\\forestfires.csv", sep=",", header=T)

forestfiresData<-na.omit(forestfiresData)

# forecast OLS 

frml <- lm ( formula = area ~ temp + wind + rain, data= forestfiresData)


# predictedValues: vector of values of the dependent variable predicted by the model

predictedValues <- exp(predict(frml))
predictedValues <- as.vector (predictedValues)

# actualValues: vector of actual values of the dependent variable

actualValues <- forestfiresData$area

#Function that calculates and returns the Root Mean Squared Error-RMSE

calculateRMSE<-function(predictedValues, actualValues){
  err<- sqrt( mean((actualValues - predictedValues)^2)  )
  return( err )
}


#10-Fold Cross Validation

# Function Parameters:
# forestfiresData: the data set to be divided into testData and trainData 
# frml: the linear regression model whose predictive accuracy will be assessed
# 10 : it's the parts which will be separated the original data set
k<-10
kFoldCrossValidation<-function(forestfiresData, frml, k){
  
# Randomly shuffle the observations of the dataset
  dataset<-forestfiresData[sample(nrow(forestfiresData)),]
  
#Generate k in number parts of the data set with approximately equal number of observations in each segment.
  folds <- cut(seq(1,nrow(dataset)), breaks=10, labels=FALSE)

  RMSE<-vector()
  
#Iterative process where each of the 10 segments will be used sequentially as the control set for the regression 
#model and all the rest as the training set.The process will terminate if all sections have been used as a check set.
 
   for(i in 1:10){
# Define the control part for the current iteration 
    testIndexes <- which(folds==i,arr.ind=TRUE)
# Define control set of the model
    testData <- dataset[testIndexes, ]
# train data, all other than the data used for control
    trainData <- dataset[-testIndexes, ]
# Estimate coefficients ocandidate.linear.model<-lm( frml, data = trainData)f the regression model using the training set
    candidate.linear.model<-lm( frml, data = trainData)
# Calculate the values of the dependent variable predicted by the model for the current control set values
    predicted<-predict(candidate.linear.model, testData)
# Calculate RMSE
    error<-calculateRMSE(predicted, testData[, "area"])
    RMSE<-c(RMSE, error)
  }
# Return average value of the errors that occurred from all control parts
  return( mean(RMSE) )
}

#RMSE

predictionModel<-vector()
predictionModel[1]<-"area ~ temp + wind + rain"
modelMeanRMSE<-vector()

for (k in 1:length(predictionModel)){
# 10-fold cross-validation for the linear regression model k
  modelErr<-kFoldCrossValidation(forestfiresData, as.formula(predictionModel[k]), 10)
  
  modelMeanRMSE<-c(modelMeanRMSE, modelErr)
  print( sprintf("Linear regression model [%s]: prediction error [%f]", predictionModel[k], modelErr ) )
  modelMeanRMSE
  }

#modelMeanRMSE




#Second question


kFoldCross32Validation<-function(forestfiresData, frml, k){
  
  # Randomly shuffle the observations of the dataset
  dataset <-  forestfiresData[ which(forestfiresData[, "area"] < 3.2), ]
  
  #Generate k in number parts of the data set with approximately equal number of observations in each segment.
  folds <- cut(seq(1,nrow(dataset)), breaks=10, labels=FALSE)
  
  RMSE<-vector()
  
  #Iterative process where each of the 10 segments will be used sequentially as the control set for the regression 
  #model and all the rest as the training set.The process will terminate if all sections have been used as a check set.
  
  for(i in 1:10){
    # Define the control part for the current iteration 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # Define control set of the model
    testData <- dataset[testIndexes, ]
    # train data, all other than the data used for control
    trainData <- dataset[-testIndexes, ]
    # Estimate coefficients of the regression model using the training set
    candidate.linear.model<-lm( frml, data = trainData)
    # Calculate the values of the dependent variable predicted by the model for the current control set values
    predicted<-predict(candidate.linear.model, testData)
    # Calculate RMSE
    error<-calculateRMSE(predicted, testData[, "area"])
    RMSE<-c(RMSE, error)
  }
  # Return average value of the errors that occurred from all control parts
  return( mean(RMSE) )
}

#RMSE

predictionModel<-vector()
predictionModel[1]<-"area ~ temp + wind + rain"

modelMeanRMSE<-vector()


for (k in 1:length(predictionModel)){
  # 10-fold cross-validation for the linear regression model k
  modelErr<-kFoldCross32Validation(forestfiresData, as.formula(predictionModel[k]), 10)
  
  modelMeanRMSE<-c(modelMeanRMSE, modelErr)
  print( sprintf("Linear regression model [%s]: prediction error [%f]", predictionModel[k], modelErr ) )
}


modelMeanRMSE


#The model with the lowest mean squared error

bestModelIndex<-which( modelMeanRMSE == min(modelMeanRMSE) )

#The model with the smallest mean square error (the highest accuracy)

print( sprintf("Model with best accuracy was: [%s] error: [%f]", predictionModel[bestModelIndex], modelMeanRMSE[bestModelIndex]) )

#For the model with the lowest mean error, its coefficients are estimated
# considering the entire data set as the training set

final.linear.model<-lm( as.formula(predictionModel[bestModelIndex]), data=forestfiresData )

final.linear.model
