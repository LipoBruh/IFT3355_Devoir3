import pandas as pd
import numpy as np 
#reference datasets
from sklearn.datasets import load_wine
#models
from sklearn.model_selection import train_test_split
#
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Docs : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine

#Part1.1
wine = load_wine(as_frame=True) #dictionnary containing the data, we can use args to extract attributes
data = wine.data
target = wine.target


# Reference : https://www.alldatascience.com/classification/wine-dataset-analysis-with-python/
#Part1.2
#
#summary as
print('\n\n\n')
#print(wine.DESCR) also informs of the data
print(data.describe()) ##Possible because data is a pandas Frame because of `load_wine(as_frame=True)`
print('\n\n\n')




##