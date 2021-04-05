# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Charger le jeu de données
dataset = pd.read_csv('qualite-vin-rouge.csv')

# Le parametre Qualite est qu'on doit prédire est représenté par y
y = dataset[['qualité']]

# Tous les parametres d'entrée utilisés pour prédire la valeur sont représentés par X
X = dataset[['acidité fixe', 'acidité volatile', 'acide citrique', 'sucre résiduel',
             'chlorures', 'dioxyde de soufre libre', 'anhydride sulfureux total',
             'densité', 'pH', 'sulphates', 'alcool']]
dataset.columns

# On frationne notre dataset en données train et en données test
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, random_state=0)

# Regression Polynomiale 
model = PolynomialFeatures(degree=4)
poly_features = model.fit_transform(X)
poly_features_test = model.fit_transform(X_test)

Lg = LinearRegression()
Lg.fit(poly_features, y)
Donnees_predites = Lg.predict(poly_features_test)

#============ Evaluer le model ================
print(mean_squared_error(y_test, Donnees_predites))

#Model 2 : degre 5
model_2 = PolynomialFeatures(degree=5)
poly_features_2 = model_2.fit_transform(X)

poly_features_test_2 = model_2.fit_transform(X_test)

Lg_2 = LinearRegression()
Lg_2.fit(poly_features_2, y)

Donnes_pred_2 = Lg_2.predict(poly_features_test_2)
print(mean_squared_error(y_test, Donnes_pred_2))    # --> 0.00026528446531238646

"""
    la comparaison des erreurs quadratiques moyennes des degres 4 et 5 
    montre qu'on a une meilleure performance avec le degre 5
    parce c'est avec lui qu'on a une plus petite ecart
"""


