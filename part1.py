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




FILE = "log.txt"

def clear_file():
    with open(FILE, 'w'):
        pass  

def append_to_file( text):
    with open(FILE, 'a') as file:
        file.write(text + '\n')

def append_frame_to_file( frame):
    with open(FILE, 'a') as f:
        f.write(frame.to_string(index=False))
        f.write('\n\n')

def append_array_to_file(array):
    with open(FILE, 'a') as f:
        f.write(str(array))
        f.write('\n\n')



print('...START...')




clear_file()



#Part1.1.2
wine = load_wine(as_frame=True) #dictionnary containing the data, we can use args to extract attributes
data = wine.data
target = wine.target


# Reference : https://www.alldatascience.com/classification/wine-dataset-analysis-with-python/
#Part1.1.2
#
#summary as
append_to_file('\n\n\n1.1 Head of Wine DataSet')
#print(wine.DESCR) also informs of the data
append_frame_to_file(data.describe()) ##Possible because data is a pandas Frame because of `load_wine(as_frame=True)`
#print('\n\n\n')




# Good reference used : https://www.youtube.com/watch?v=0B5eIE_1vpU
# Split : https://realpython.com/train-test-split-python-data/
## Part 1.2.1

# Unecessary because GridSerchCV has a parameter for that
X,y = load_wine(return_X_y=True)
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Part 1.2.2 - Models without preprocessing


#1
model_DT = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid={},
    cv=5            #1.2.1 : 80% training, 20% testing
    ).fit(X,y)
#2
model_RF = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={},
    cv=5
    ).fit(X,y)
#3
model_SVM = GridSearchCV(
    estimator=svm.SVC(), #SVC is the regressor
    param_grid={},
    cv=5
    ).fit(X,y)
#4
model_MLP = GridSearchCV(
    estimator=MLPRegressor(),
    param_grid={},
    cv=5
    ).fit(X,y)

## Part 1.2.3 - Metrics - We are looking for mean test score
append_to_file("\n\n")
append_to_file("1.2 TESTS - NO PREPROCESSING : ACCURACY = mean_test_score\n")
#1
append_frame_to_file(pd.DataFrame(model_DT.cv_results_))
#2
append_frame_to_file(pd.DataFrame(model_RF.cv_results_))
#3
append_frame_to_file(pd.DataFrame(model_SVM.cv_results_))
#4
append_frame_to_file(pd.DataFrame(model_MLP.cv_results_))


append_to_file("\n\n")
append_to_file("Weight of Features of the Model\n")
append_to_file("DecisionTreeRegressor")
append_array_to_file(model_DT.best_estimator_.feature_importances_)
append_to_file("RandomForestRegressor")
append_array_to_file(model_RF.best_estimator_.feature_importances_)
#print(model_SVM.best_estimator_)
#print(model_SVM.best_estimator_.feature_importances_) #feature_importances_ do not exist for those models
#print(model_MLP.best_estimator_)
#print(model_MLP.best_estimator_.feature_importances_) #same
#print("!!!!!\n")


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
    cv=5
    ).fit(X,y)
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
    cv=5
    ).fit(X,y)
#3
pipe_SVM = Pipeline([
    ("scale",StandardScaler()),
    ("pca", PCA(n_components=4)),
    ("model",svm.SVC())
    ])
model_SVM2 = GridSearchCV(
    estimator=pipe_SVM, #SVC is the regressor
    param_grid={},
    cv=5
    ).fit(X,y)
#4
pipe_MLP = Pipeline([
    ("scale",StandardScaler()),
    ("pca", PCA(n_components=4)),
    ("model",MLPRegressor())
    ])
model_MLP2 = GridSearchCV(
    estimator=pipe_MLP,
    param_grid={},
    cv=5
    ).fit(X,y)

## Part 1.2.3 - Metrics - We are looking for mean test score
append_to_file("\n\n")
append_to_file("1.3 - TESTS - WITH PREPROCESSING : ACCURACY = mean_test_score\n")
#1
append_frame_to_file(pd.DataFrame(model_DT2.cv_results_))
#2
append_frame_to_file(pd.DataFrame(model_RF2.cv_results_))
#3
append_frame_to_file(pd.DataFrame(model_SVM2.cv_results_))
#4
append_frame_to_file(pd.DataFrame(model_MLP2.cv_results_))

print('...DONE...')