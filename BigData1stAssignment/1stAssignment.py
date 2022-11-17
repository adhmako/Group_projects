##########  Erwthma 3 ###########
print("Erwthma 3")

import pandas as pd
from sklearn import decomposition
from sklearn.decomposition import PCA
import csv

#read csv
data=pd.read_csv("/Users/dhmako/Downloads/winequalitywhite.csv", sep=";", header=0, decimal=".")


print(data.head())

pca = decomposition.PCA(n_components=4)
pca.fit(data)

print("Eigenvectors:")
print(pca.components_)

transformeddata = pca.transform(data)

print("\nNew data points, on the space defined by the Eigenvectors")
print( transformeddata)
print("\n Variance")
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

##########  Erwthma 4a ###########
print("Erwthma 4a")

import numpy as np
import pandas as pd
import math

#%erwthma1
#a
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,4,5,6])
print("Euclidean distance between x and y is: ")
eucldist = math.sqrt(sum((x-y)**2))
print(eucldist)

#b
x=np.array([-0.5,1,7.3,7,9.4,-8.2,9,-6,-6.3])
y=np.array([0.5,-1,-7.3,-7,-9.4,8.2,-9,6,6.3])
print("Euclidean distance between x and y is: ")
eucldist = math.sqrt(sum((x-y)**2) )
print(eucldist)

#c
x=np.array([-0.5,1,7.3,7,9.4,-8.2])
y=np.array([1.25,9.02,-7.3,-7,5,1.3])
print("Euclidean distance between x and y is: ")
eucldist = math.sqrt(sum((x-y)**2) )
print(eucldist)

#d
x=np.array([0,0,0.2])
y=np.array([0.2,0.2,0])
print("Euclidean distance between x and y is: ")
eucldist = math.sqrt(sum((x-y)**2) )
print(eucldist)

##########  Erwthma 4b ###########
print("Erwthma 4b")

x=np.array([25000,14,7])
y=np.array([42000,17,9])
z=np.array([55000,22,5])
n=np.array([27000,13,11])
m=np.array([58000,21,13]) #target of comparison

dist=np.array([(math.sqrt(sum((x-m)**2))),(math.sqrt(sum((y-m)**2))),(math.sqrt(sum((z-m)**2))),(math.sqrt(sum((n-m)**2)))])

print("\n The user that is the most similiar to user with id=5 is")

print(pd.Series(dist).idxmin() + 1)

##########  Erwthma 5 ###########
print("Erwthma 5")

import numpy as np
from numpy.linalg import norm


A = np.array([9.32, -8.3, 0.2])
B = np.array([-5.3, 8.2, 7])

cosine = np.dot(A,B)/(norm(A)*norm(B))

print("Cosine Similarity:", cosine)

A= np.array([6.5, 1.3, 0.3, 16, 2.4, -5.2, 2, -6, -6.3])
B= np.array([0.5, -1, -7.3, -7, -9.4, 8.2, -9, 6, 6.3])

cosine = np.dot(A,B)/(norm(A)*norm(B))

print("Cosine Similarity:", cosine)

A= np.array([-0.5, 1, 7.3, 7, 9.4, -8.2])
B= np.array([1.25, 9.02, -7.3, -7, 15, 12.3])

cosine = np.dot(A,B)/(norm(A)*norm(B))

print("Cosine Similarity:", cosine)

A= np.array([2, 8, 5.2])
B= np.array([2, 8, 5.2])

cosine = np.dot(A,B)/(norm(A)*norm(B))

print("Cosine Similarity:", cosine)

##########  Erwthma 6 ###########
print("Erwthma 6")

import numpy as np

x = np.array(["Green", "Potato", "Ford"])
y = np.array(["Tyrian purple", "Pasta", "Opel"])

nominaldis = np.sum(x!=y)
print(" The distance between the nominal arrays is" , nominaldis)     

x = np.array(["Eagle", "Ronaldo", "Real Madrid", "Prussian blue", "Michael Bay"])
y = np.array(["Eagle", "Ronaldo", "Real Madrid", "Prussian blue", "Michael Bay"])

nominaldis = np.sum(x!=y)
print(" The distance between the nominal arrays is" , nominaldis)             

x = np.array (["Werner Herzog", "Aquirre,the wrath of God", "Audi", "Spanish red"])
y = np.array (["Martin Scorsese", "Taxi driver", "Toyota", "Spanish red"])

nominaldis = np.sum(x!=y)
print(" The distance between the nominal arrays is" , nominaldis)    


