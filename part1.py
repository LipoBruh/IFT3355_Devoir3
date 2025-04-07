import pandas as pd
import numpy as np 
#reference datasets
from sklearn.datasets import load_wine
#models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
#Preprocessing
from sklearn.preprocessing import StandardScaler #for normalizing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#Metrics
from sklearn.metrics import accuracy_score

#Docs : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine

#Part1.1.2
wine = load_wine(as_frame=True) #dictionnary containing the data, we can use args to extract attributes
data = wine.data
target = wine.target


# Reference : https://www.alldatascience.com/classification/wine-dataset-analysis-with-python/
#Part1.1.2
#
#summary as
print('\n\n\n')
#print(wine.DESCR) also informs of the data
print(data.describe()) ##Possible because data is a pandas Frame because of `load_wine(as_frame=True)`
print('\n\n\n')




# Good reference used : https://www.youtube.com/watch?v=0B5eIE_1vpU
# Split : https://realpython.com/train-test-split-python-data/
## Part 1.2.1
X,y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Part 1.2.2 - Models without preprocessing


#1
model_DT = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid={},
    ).fit(X,y)
#2
model_RF = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={},
    ).fit(X,y)
#3
model_SVM = GridSearchCV(
    estimator=svm.SVC(), #SVC is the regressor
    param_grid={},
    ).fit(X,y)
#4
model_MLP = GridSearchCV(
    estimator=MLPRegressor(),
    param_grid={},
    ).fit(X,y)

## Part 1.2.3 - Metrics - We are looking for mean test score
print("\n\n")
print("TESTS - NO PREPROCESSING : ACCURACY = mean_test_score\n")
#1
print(pd.DataFrame(model_DT.cv_results_))
#2
print(pd.DataFrame(model_RF.cv_results_))
#3
print(pd.DataFrame(model_SVM.cv_results_))
#4
print(pd.DataFrame(model_MLP.cv_results_))

