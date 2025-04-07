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
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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
    ).fit(x_train,y_train)
#2
model_RF = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={},
    ).fit(x_train,y_train)
#3
model_SVM = GridSearchCV(
    estimator=svm.SVC(), #SVC is the regressor
    param_grid={},
    ).fit(x_train,y_train)
#4
model_MLP = GridSearchCV(
    estimator=MLPRegressor(),
    param_grid={},
    ).fit(x_train,y_train)

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


print("!!!!!\n")
print(model_DT.best_estimator_)
print(model_DT.best_estimator_.feature_importances_)
print(model_RF.best_estimator_)
print(model_RF.best_estimator_.feature_importances_)
#print(model_SVM.best_estimator_)
#print(model_SVM.best_estimator_.feature_importances_) #feature_importances_ do not exist for those models
#print(model_MLP.best_estimator_)
#print(model_MLP.best_estimator_.feature_importances_) #same
print("!!!!!\n")


## Part 1.3.1
# StandardScaler for scaling for mean around 0 and sd around 1
# PCA for reducing the model size
# `PCA` will transform our 13 features into 4 new ones.



#1
pipe_DT = Pipeline([
    ("scale",StandardScaler()), ## Part 1.3.1
    ("pca", PCA(n_components=4)), ## Part 1.3.2
    ("model",DecisionTreeRegressor())
    ])
#
model_DT2 = GridSearchCV(
    estimator=pipe_DT,
    param_grid={},
    ).fit(x_train,y_train)
#2
pipe_RF = Pipeline([
    ("scale",StandardScaler()),
    ("pca", PCA(n_components=4)),
    ("model",RandomForestRegressor())
    ])
#
model_RF2 = GridSearchCV(
    estimator=pipe_RF,
    param_grid={},
    ).fit(x_train,y_train)
#3
pipe_SVM = Pipeline([
    ("scale",StandardScaler()),
    ("pca", PCA(n_components=4)),
    ("model",svm.SVC())
    ])
model_SVM2 = GridSearchCV(
    estimator=pipe_SVM, #SVC is the regressor
    param_grid={},
    ).fit(x_train,y_train)
#4
pipe_MLP = Pipeline([
    ("scale",StandardScaler()),
    ("pca", PCA(n_components=4)),
    ("model",MLPRegressor())
    ])
model_MLP2 = GridSearchCV(
    estimator=pipe_MLP,
    param_grid={},
    ).fit(x_train,y_train)

## Part 1.2.3 - Metrics - We are looking for mean test score
print("\n\n")
print("TESTS - WITH PREPROCESSING : ACCURACY = mean_test_score\n")
#1
print(pd.DataFrame(model_DT2.cv_results_))
#2
print(pd.DataFrame(model_RF2.cv_results_))
#3
print(pd.DataFrame(model_SVM2.cv_results_))
#4
print(pd.DataFrame(model_MLP2.cv_results_))

