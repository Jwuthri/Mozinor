
# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, RidgeCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor, XGBClassifier

from vecstack import stacking

# Read the csv file
data = pd.read_csv("toto.csv")
regression = False
if regression:
    metric = r2_score
else:
    metric = accuracy_score

# Split dependants and independant variables
y = data[["predict"]]
X = data.drop("predict", axis=1)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Apply Some Featuring
poly_reg = PolynomialFeatures(degree=1)

# Transform into numpy object
x_train = poly_reg.fit_transform(X_train)
x_test = poly_reg.fit_transform(X_test)
y_test = np.array(y_test.ix[:,0])
y_train = np.array(y_train.ix[:,0])

# define lmodels
lmodels = [ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
           max_depth=None, max_features=0.6, max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=4, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False), XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.5, max_delta_step=0, max_depth=8,
       min_child_weight=6, missing=None, n_estimators=50, nthread=-1,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.9), KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=17, p=2,
           weights='distance')]

# build the stack level 1
S_train, S_test = stacking(
    lmodels, x_train, y_train, x_test,
    regression=regression, metric=metric,
    n_folds=3, shuffle=True, random_state=0, verbose=1
)

# build model lvel 2
model = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=2,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

# Fit the model
model.fit(S_train, y_train)

# Predict
y_pred = model.predict(S_test)

# Scoring
if regression:
    print('Score on test set:', mean_absolute_error(y_test, y_pred))
else:
    print('Score on test set:', accuracy_score(y_test, y_pred))
print(metric(y_test, y_pred))

