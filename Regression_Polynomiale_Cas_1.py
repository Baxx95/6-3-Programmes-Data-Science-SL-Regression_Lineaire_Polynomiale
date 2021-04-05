# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:07:22 2021

@author: Zakaria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# CHARGER LE JEU DE DONNEES
dataset = pd.read_csv("Position_Salaries.csv")

# on recupere la colonne niveau
X = dataset.iloc[:, 1:2].values

# On recupere le salaire
Y = dataset.iloc[:, 2:3].values 
# ou dataset.iloc[:, -1].values --> parce que c'est la dernier colonne de notre daataset


#===== Ajout de polynôme pour un meilleur ajustement des données =====
# Regression Polynomiale
model = PolynomialFeatures(degree=2)
model_1 = PolynomialFeatures(degree=3)
model_2 = PolynomialFeatures(degree=4)

X_poly_features = model.fit_transform(X)
X_poly_features_1 = model_1.fit_transform(X)
X_poly_features_2 = model_2.fit_transform(X)

model.fit(X_poly_features, Y)
model_1.fit(X, Y)
model_2.fit(X, Y)

# Regresson Polynômiale
poly_regression = LinearRegression()
poly_regression_1 = LinearRegression()
poly_regression_2 = LinearRegression()
poly_regression.fit(X_poly_features, Y)
poly_regression_1.fit(X_poly_features_1, Y)
poly_regression_2.fit(X_poly_features_2, Y)

# Regression normale
LinReg = LinearRegression()
LinReg.fit(X, Y)

# Visualisation regression Polynomiale
plt.scatter(X, Y)
plt.plot(X, poly_regression.predict(X_poly_features))
plt.title("Regression Polyomiale : Experience par rapport au Salaire 2")
plt.xlabel("Experience")
plt.ylabel("Salaire")
plt.show()

# Visualisation regression Lineaire
plt.scatter(X, Y)
plt.plot(X, LinReg.predict(X))
plt.title("Regression Lineaire : Experience par rapport au Salaire 2")
plt.xlabel("Experience")
plt.ylabel("Salaire")
plt.show()

#Visualisation des deux regression sur le même graphe
plt.scatter(X, Y)

plt.plot(X, LinReg.predict(X))
plt.plot(X, poly_regression.predict(X_poly_features))
plt.plot(X,poly_regression_1.predict(X_poly_features_1))
plt.plot(X,poly_regression_2.predict(X_poly_features_2))

plt.legend()
plt.title("Regression Lineaire et Polynomiale : Experience par rapport au Salaire 2")
plt.xlabel("Experience")
plt.ylabel("Salaire")
plt.show()


