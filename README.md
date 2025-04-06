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





## Notes on SciKit-Learn

#### Data

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

#### Models

1. Create the model as a python object
2. Make the model 'learn' with `.fit(X,Y)`




## Scikit Learn Wine

#### Methods :

```py
#Imports
from sklearn import datasets
wine = datasets.load_wine()


#Log the dataset description
print(wine.DESCR)

#Print the head of the attributes
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['label'] = wine.target
print(df.head())


```



## Part 1 

### Section 1
> Chargez le dataset Wine avec sklearn.datasets.load_wine().
> Affichez un résumé statistique des données : moyenne et écart-type de chaque caractéristiques et ce pour chaque classe (dans un tableau).



### Section 2
> Séparez les données en un ensemble d’entraînement (80%) et un ensemble de test (20%) avec train_test_split.
> Testez les modèles suivants avec leurs paramètres par défaut : Arbre de décision, Random Forest, SVM, Perceptron multicouche (MLP).
> Calculez l’accuracy sur l’ensemble de test pour chaque modèle et présentez les résultats dans un tableau.



### Section 3
> Appliquez une normalisation (ex. StandardScaler) pour centrer et réduire les données.
> Réduisez les dimensions avec PCA (gardez 4 composantes principales).
> Retestez les mêmes modèles sur les données prétraitées et comparez les résultats avec ceux sans prétraitement (tableau ou graphique).



### Section 4
> Commentez brièvement l’impact de la normalisation et de PCA sur les performances des modèles.


