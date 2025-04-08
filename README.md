# IFT3355_Devoir3
AI Homework on scikit-learn


## Contributors :

- Emanuel Rollin - 20106951
- Anne Sophie Rozefort - 20189221





## References
- [Scikit-learn.org](https://scikit-learn.org/stable/)
- [Documentation](https://scikit-learn.org/0.21/documentation.html)
- [Wine Scikit-Learn Tutorial](https://www.youtube.com/watch?v=rcMTeiVySV8)
- [2h Scikit Tutorial](https://www.youtube.com/watch?v=0B5eIE_1vpU)





## Prerequisites 

Make sure you have python installed globally. Else not much will work lol.
Pip is usefull, worth having too.


###### Linux : 
```bash
sudo apt update
sudo apt install python3
#You usually need those too : 
sudo apt install python3-venv python3-full
```

###### Windows :
[Releases & Installers](https://www.python.org/downloads/windows/)





## Dependencies

How to install the dependencies from requirement.txt :

Create a virtual environment in your project directory:

```bash
python -m venv venv
#then start a terminal in your virtual environment 
```

```bash
#Activate your virtual environment

#Windows :
.\venv\Scripts\activate
#Linux : 
source venv/bin/activate
```

Make sure the environment is selected. Use the shortcut Ctrl+Shift+P and pick the Python: Select Interpreter option and select the venv interpreter. Then you can install the `requirements.txt` without impacting your global dependencies.


And then install them with :

```bash
pip install -r requirements.txt
```

That's it !




## Part 1 - Rapport

### Section 1
> Chargez le dataset Wine avec sklearn.datasets.load_wine().
> Affichez un résumé statistique des données : moyenne et écart-type de chaque caractéristiques et ce pour chaque classe (dans un tableau).

Check `part1.py` for the implementation and `log.txt` for the output

### Section 2
> Séparez les données en un ensemble d’entraînement (80%) et un ensemble de test (20%) avec train_test_split.
> Testez les modèles suivants avec leurs paramètres par défaut : Arbre de décision, Random Forest, SVM, Perceptron multicouche (MLP).
> Calculez l’accuracy sur l’ensemble de test pour chaque modèle et présentez les résultats dans un tableau.

Because we are using `GridSearchCV`, we can specify `cv=5` to use 1 part for testing and 4 parts for training, without using `train_test_split`. The line that uses the method was commented out.

```py
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_DT = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid={},
    cv=5            #1.2.1 : 80% training, 20% testing
    ).fit(X,y)
```

To observe the accuracy, we can use the property of `GridSearchCV` that gives us the mean test score, using the attribute `cv_results_`.

```py
#look for the column `mean_test_score`
print(pd.DataFrame(model_DT.cv_results_))
```


### Section 3
> Appliquez une normalisation (ex. StandardScaler) pour centrer et réduire les données.
> Réduisez les dimensions avec PCA (gardez 4 composantes principales).
> Retestez les mêmes modèles sur les données prétraitées et comparez les résultats avec ceux sans prétraitement (tableau ou graphique).

`print(model_DT.best_estimator_.feature_importances_)`

By using this line, we can find the best model produced by `GridSearchCV` and return an array of the columns / classes / features used to train the model. Columns with `0.` are not used, others are. 

For `DecisionTreeRegressor` and `RandomForestRegressor` both give a lot of importance to these features:


- 56-65% : Flavanoids (7)
- 5 - 8% : Color intensity (10)
- 1%     : Hue (11)
- 24-27% : Proline (13)

We could expect that the PCA filtering to, when asked to reduce their features to 4, give the most importance to those four and correlate the rest to them.


### Section 4
> Commentez brièvement l’impact de la normalisation et de PCA sur les performances des modèles.

Consult log.txt to have the metrics (accuracy) produced by `GridSearchCV`.





| Model                 | Accuracy (R2) | Accuracy w/ preprocessing |
| :-------------------- | :-----------: | :------: |
| DecisionTreeRegressor |  0.844        | 0.845    |
| RandomForestRegressor |  0.897        | 0.893    |
| svm.SVC               |  0.633        | 0.979    |
| MLPRegressor          |  -152         | 0.889    |


#### Scaling
[Reference1](https://www.youtube.com/watch?v=6eJHk8JYK2M)
[Referemce2](https://www.youtube.com/watch?v=sxEqtjLC0aM)

Normalization is : x-min(x) / (max(x)-min(x)) -> Bounded between 0 and 1

Standardization is : x-mean(x)/standard deviation -> Mean centered around 0 with values above and below


The scaling should have litte to no impact on tree models. We know from the [Docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) that both `DecisionTreeRegressor` and `RandomForestRegressor` are tree models, so a lack of significative change from preprocessing makes sense. Treebased algorithms use relative comparisons where only the ordering of values is important.

Algorithms that requires calculation of distances between points often benefit from scaling, to avoid favoring one axis over the other (`svm.SVC`). Scaling also avoids oscillation during the gradient descent which favors `MLP` that uses backpropagation.



