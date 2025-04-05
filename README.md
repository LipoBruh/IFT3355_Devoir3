# IFT3355_Devoir3
AI Homework on scikit-learn


## Contributors :

- Emanuel Rollin - 20106951
- Anne Sophie Rozefort - 20189221





## References
- [Scikit-learn.org](https://scikit-learn.org/stable/)
- [Documentation](https://scikit-learn.org/0.21/documentation.html)
- [Wine Scikit-Learn Tutorial](https://www.youtube.com/watch?v=rcMTeiVySV8)





## Prerequisites 

Make sure you have python installed globally. Else not much will work lol.
Pip is usefull, worth having too.

###### Linux : 
```bash
sudo apt update
sudo apt install python3
```

###### Windows :
[Releases & Installers](https://www.python.org/downloads/windows/)





## Dependencies

How to install the dependencies from requirement.txt :

Create a virtual environment in your project directory:

```bash
python -m venv venv
#then
.\venv\Scripts\activate
```

And then install them with :

```bash
pip install -r requirements.txt
```

Make sure the environment is selected. Use the shortcut Ctrl+Shift+P and pick the Python: Select Interpreter option and select the venv interpreter.






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


