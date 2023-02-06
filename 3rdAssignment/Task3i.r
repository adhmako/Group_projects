# install.packages("rpart")
# install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

setwd("C:/Master/BD/Projects/Project3")

# Reading the mushroom data
musdata <- read.table("agaricuslepiota.data", sep=",", header=FALSE)
musdata[, 1] <- factor(musdata[, 1], levels = c("e", "p"), labels = c("edible", "poisonous"))
colnames(musdata)[1] <- "classes"

#write.csv(musdata, file = "musdata.csv", row.names = FALSE) --> if i want to export this dataset

# Splitting the data into training and testing sets
set.seed(123)
splitindex <- sample(1:nrow(musdata), size = 0.8*nrow(musdata))
training <- musdata[splitindex,]
testing <- musdata[-splitindex,]

# Creating and visualizing the decision tree
treemodel <- rpart(classes ~ ., data = training, method = "class")
rpart.plot(treemodel, extra = 104)

# Predictions using the model
predictions <- predict(treemodel, newdata = testing, type = "class")
print(predictions)

# Confusion Matrix 
confmatrix <- table(predictions, testing$classes)
print(confmatrix)

# Accuracy
accuracy <- sum(diag(confmatrix))/sum(confmatrix)
print(accuracy)

