

## Notes on SciKit-Learn

### Data

- X : Info about our entities
- Y : Entities with known attributes

You can import a reference dataset with :
```py
from sklearn.datasets import dataset_name
```

You should be able to extact and manipulate the dataset using `load_datasetName` or similar functions. They return a dictionnary, you can filter the data by provinding arguments to the `load` function.

The arguments are not standardized, but common ones are :

```py
load_datasetName(return_X_y=True)
#return two arrays, one for X data, another for Y data
load_datasetName(as_frame=True)
#useful for NumPy
```

### Models

1. Create the model as a python object

2. Make the model 'learn' with `.fit(X,Y)`



#### Linear Regression
```py
#Many models are available, here we use the linear regression
mod = LinearRegression()

mod.fit(X,y)
```

Then, you can use the model to make predictions on entities :

```py
mod.predict(X) #array of predictions
```

You can visualize the accuracy of the model with a scatterplot : 
```py
plt.scatter(mod.predict(X),y)
```


#### K Neighbors Regressor

If the data is not normalized, the corrolation is more noisy and may be distributed in stange ways (one axis stretched over the other). We usually normalize X, aka the values of the attributes of the entities. With normalization :

```py
mod = KNeighborsRegressor().fit(X,y)
pipe = Pipeline([
    ("scale",StandardScaler()),
    ("model",KNeighborsRegressor())
    ])
pipe.fit(X,y)
pipe.predict(X)
```


It stills needs reworking because we include the original data to make predictions. We can use different subsets of the data to train and others to predict. The TP calls for data to be split in a 20% to predict and 80% to train ratio. We can use `GridSearchCV` to help that process.

```py

pipe = Pipeline([
    ("scale",StandardScaler()),
    ("model",KNeighborsRegressor())
    ])

#pipe.get_params() ->n_neighbors
mod = GridSearchCV(
    estimator=pipe,
    param_grid={
        'model_n_neighbors':[1,2,...,10],
        },
    cv=3, #cross validation
)
mod.fit(X,y)
print(pd.DataFrame(mod.cv_results_))

```


#### Isolation Forest
The model `IsolationForest` finds outliers in a data set. The model will return -1 if an outlier is found and 1 if not.

```py
from collections import Counter
from skleanr.ensemble import IsolationForest

mod = IsolationForest().fit(X)
nmod.predict(x)
```





### Preprocessing 

Using `StandardScaler`, we can normalize. Standard Scaler will arrange the data round the value 0 using standard deviation.

```py
from sklean.preprocessing import StandardScaler

X_new = StandardScaler().fit_transform(X)
```



Using `QuantileTransformer`, we can arrange the data by quantiles. Will affect how the model works. Reduces he influence of the outliers.

```py
from sklean.preprocessing import QuantileTransformer

X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
```



Using the `PolynomialFeatures()` filter helps filtering data when they are not linearly divided. 

```py
from sklearn.preprocessing import PolynomialFreatures.
```



Using `OneHotEncoder`, we can preprocess classes, categories, strings... and it will be converted into numerical values that can be used. 

```py
from sklearn.preprocessing import OneHotEncoder
#
arr = [['low','medium','low','high']]
enc=OneHotEncoder(sparse=False, handle_unknown='ignore') #ignore data X that was never trained on
enc.fit_transform(arr)
```



We would simply have to change our preprocessing algorithm in the pipeline.

```py
pipe = Pipeline([
    ("scale",Preprocess()),
    ("model",Model())
    ])
```





### Metrics



#### Accuracy
Used to compare models. The built in metric of a model is accuracy (`model.score`), but other metrics can be used.

```py
grid = GridSearchCV(
    estimator=LogisticRegression(
        class_weight={0:1,1:2}, max_iter=1000
        ),
    param_grid={
        'model_n_neighbors':[1,2,...,10],
        },
    cv=4, #cross validation
    n_jobs= -1, #parallel training
    )
grid.fit(X,y)
#will print the scores for the 10 models we made
#(because of model_n_neighbors)
print(pd.DataFrame(mod.cv_results_))
```




#### Sum of positives
```py
mod = LogisticRegression(class_weight={0:1,1:2}, max_iter=1000)
#class weight changes how much importance we give to one category of X so that it has more importance for that model

mod.fit(X,y).predict(X).sum()
#will return the number of positives?
```



#### Score

```py
from sklearn.metrics. import precision_score, recall_score

#when we predict "class", how accurate are we (false negatives)
precision_score(y, grid.predict(X))

#% of positives (false positives)
recall_score(y, grid.predict(X))
```


#### Reoptimize with a metric

```py
grid = GridSearchCV(
    estimator=LogisticRegression(
        class_weight={0:1,1:2}, max_iter=1000
        ),
    param_grid={
        'model_n_neighbors':[1,2,...,10],
        },
    cv=4, #cross validation
    n_jobs= -1, #parallel training
    scoring={
        'precision':make_scorer(precision_score),
        'recall_store':make_scorer(recall_score),
        }#visible scores in results
    refit='precision', #optimize models
    return_train_score=True,
    )
grid.fit(X,y)
#will print the scores for the 10 models we made
#(because of model_n_neighbors)
print(pd.DataFrame(mod.cv_results_))
```

#### Balance between Recall and Precision

We can visualize the metrics with a plot, to observe the score from the test results or train sets. If we train many models with different `class_weight={...}`, we can plot the `recall` and `precision` score and choose a maximum balacing the two (f1 score) with :

```py
#plot
plt.figure(figsize=(12,4))
#array
df = pd.DataFrame(grid.cv_results_)
#
for score in ['mean_test_recall','mean_test_precision']:
    plt.plot(
        [_ [1] for _ in df ['param_class_weight']],
        df[score],
        label=score)
plt.legend()
```



#### Make_Scorer

The `make_scorer` function can be used in our `GridSearchCV` instanciation to convert our custom metric into a callable new object.

The metric function takes `y_true` and `y_predicted`.

The callable object takes `estimator`,`X`,`y` and `sample_weight`.

`y_predicted` can be obtained from the estimator with `est.predict(X)`. The parameter `sample_weight` can give more importance to a single `class` parameter to have more importance in your dataset. 
