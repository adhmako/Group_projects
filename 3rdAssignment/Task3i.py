import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# load the dataset
mushroom_data = pd.read_table("agaricuslepiota.data", header = None, delimiter = ',')

# column names
column_names = ['edibility', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
mushroom_data.columns = column_names
print (mushroom_data.columns)
print (mushroom_data.head())

# convert categorical variables to numerical variables with One-hot encoding.
# a list of the names of our categorical variables.
categoricalVariables=mushroom_data.select_dtypes([object]).columns

for var in categoricalVariables:
    
    print("\tOne-Hot-Encoding variable ",var, " .....", sep="", end="")
    if var == "edibility":
       print("Ignored") 
       continue 
    mushroom_data[var]=pd.Categorical(mushroom_data[var])
    varDummies = pd.get_dummies(mushroom_data[var], prefix = var)
    mushroom_data = pd.concat([mushroom_data, varDummies], axis=1)
    mushroom_data=mushroom_data.drop([var], axis=1)
    print("Done")

print("\n\tVariables of DataFrame bankData after One-hot encoding:")
print("\t", mushroom_data.columns)

# We add a new column to the end of DataFrame mushroom_data named newY, containing these new values of 0 and 1.
# Once we have done this, the original column/variable y is not needed anymore and can be dropped.
mushroom_data['newY'] = ( mushroom_data['edibility'].map( {'p':0, 'e':1}) )
mushroom_data=mushroom_data.drop(['edibility'], axis=1)
print("\n\nPreprocessing done.")

# target variable
y = mushroom_data["newY"]

# feature variables
X = mushroom_data.iloc[:, :-1]

# split the dataset into training and test sets
trainingSetFeatures, testingSetFeatures, trainingSetClass, testingSetClass = train_test_split(X, y, test_size = 0.2, random_state = 0)

# build the decision tree model
model = DecisionTreeClassifier(criterion="entropy")
model_fit = model.fit(trainingSetFeatures, trainingSetClass)

# make predictions on the test set
predictions = model.predict(testingSetFeatures)

# calculate the confusion matrix and accuracy
cm = confusion_matrix(testingSetClass, predictions)
acc = accuracy_score(testingSetClass, predictions)

# print the confusion matrix and accuracy
print("Confusion matrix:")
print(cm)
print("Accuracy:", acc)

# visualize the decision tree
plt.figure(figsize=(15,5))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=["edible", "poisonous"])
plt.show()
