
# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble.forest import ExtraTreesClassifier

from vecstack import stacking

# Read the csv file
data = pd.read_csv("toto.csv")
regression = False

# Split dependants and independant variables
y = data[["predict"]]
X = data.drop("predict", axis=1)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Apply Some Featuring
poly_reg = PolynomialFeatures(degree=1)

# Transform into numpy object
x_train = poly_reg.fit_transform(X_train)
X_test = poly_reg.fit_transform(X_test)
y_test = np.array(y_test.ix[:,0])
y_train = np.array(y_train.ix[:,0])

# Build model with good params
model = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
           max_depth=None, max_features=0.6, max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=4, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

# Fit the model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Scoring
if regression:
    print('Score on test set:', mean_absolute_error(y_test, y_pred))
else:
    print('Score on test set:', accuracy_score(y_test, y_pred))

