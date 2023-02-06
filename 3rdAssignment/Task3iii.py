import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# load the dataset
mushroom_data = pd.read_table("agaricuslepiota.data", header = None, delimiter = ',')

# column names
column_names = ['edibility', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
mushroom_data.columns = column_names

# replacing any ? for missing values
mushroom_data = mushroom_data.replace('?', np.NaN)

# droppingthe missing values
mushroom_data = mushroom_data.dropna(0, how='any')

mushroom_data.info()

# Now we have to tell our dataset that the values 'y', 'n' are categorical (or factors in R parlance).
# This is called labelling in pandas. This will resule in one value 'y' to be encoded as 1 and 'n' as 0.
mushroom_data = mushroom_data.apply(preprocessing.LabelEncoder().fit_transform)

trainSet, testSet = train_test_split(mushroom_data, test_size=0.2, random_state = 0)

# trainClass will contain only the first column of the dataset (iloc enables integer-based indexing) 
trainClass = trainSet.iloc[:,0] # -> all rows (:), first column (0). IMPORTANT: Python uses 0-based indexing!

# All other attributes except first one will form our training data
trainData = trainSet.iloc[:, 1:16]

# Slice the testing set into 2 parts:  one that contains only the first row which is the class attribute and
# another one that contains only all the other variables. We do this in order to calculate the confusion matrix
# at the end
testClass = testSet.iloc[:,0]
testData = testSet.iloc[:,1:16]

# Create the Multinomial Naive Bayes classifier object to properly use a probability
mNB = MultinomialNB()
# Train the Naive Bayes classifier using the training set
NBModel = mNB.fit(trainData, trainClass)

#Predict the class attribute
testResults = NBModel.predict(testData)

# Done. Let's claculate and display the confusion matrix
cmatrix = confusion_matrix(testClass, testResults)
print(cmatrix)


# Let's see the accuracy of the Naive Bayes classifier we just built
accS = accuracy_score(testClass, testResults)
print("Accuracy score = ", accS )
