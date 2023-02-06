library(ggplot2)
setwd("//Users//dhmako/Documents//Assignments in Managing Big Data//Second Assignment")

graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

########## Linear Regression with OLS ##########

data <-read.csv("HouseholdData.csv", sep=",", header=T)

ols <- lm(FoodExpenditure ~ Income+FamilySize, data=data)

print(ols$coefficients)

########## Linear Regression with Gradient Descent ##########

calculateCost<-function(X, y, theta){
  # Πλήθος παρατηρήσεων
  m <- length(y)
  return( sum((X%*%theta- y)^2) / (2*m) )
} # calculateCost



# gradientDescent

gradientDescent<-function(X, y, theta, alpha=0.01, numIters=90){

  m <- length(y)
  
 costHistory <- rep(0, numIters)
  
 for(i in 1:numIters){
    
   theta <- theta - alpha*(1/m)*(t(X)%*%(X%*%theta - y))
    
    costHistory[i]  <- calculateCost(X, y, theta)
    
  } 
  
  gdResults<-list("coefficients"=theta, "costs"=costHistory)
  return(gdResults)
} 



y<- data[, "FoodExpenditure"]

x<- cbind( rep(1, 35), data[, "Income"], data[, "FamilySize"] ) 

initialThetas<-rep(runif(1), 3 ) 


gdOutput<-gradientDescent(x, y, initialThetas, 0.00000000003446, 65000)


# Εμφάνιση συντελεστών
print(gdOutput$coefficients)


plot(gdOutput$costs, xlab="Επαναλήψεις", ylab="J(θ)" )

