# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
from mozinor.config.params import *
from mozinor.config.explain import *


Fast_Classifiers = {
    "ExtraTreesClassifier": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "criterion": criterion,
        "max_features": max_features,
        "bootstrap": dual,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": ExtraTreesClassifier
    },
    "LogisticRegression": {
        "import": "sklearn.linear_model",
        "penalty": penalty,
        "C": penalty_factor,
        "dual": [False],
        "show": LogisticRegression
    },
    "KNeighborsClassifier": {
        "import": "sklearn.neighbors",
        "weights": weights,
        "p": [1, 2],
        "n_neighbors": [50],
        "show": KNeighborsClassifier
    }
}


Classifiers = {
    # Tree
    "DecisionTreeClassifier": {
        "import": "sklearn.tree",
        "criterion": criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": DecisionTreeClassifier
    },
    # Ensemble
    "ExtraTreesClassifier": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "criterion": criterion,
        "max_features": max_features,
        "bootstrap": dual,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": ExtraTreesClassifier
    },
    "RandomForestClassifier": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "criterion": criterion,
        "max_features": max_features,
        "bootstrap": dual,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": RandomForestClassifier
    },
    "GradientBoostingClassifier": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "learning_rate": learning_rate,
        "max_features": max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": GradientBoostingClassifier
    },
    # XGBoost
    'XGBClassifier': {
        "import": "xgboost",
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': max_features,
        'min_child_weight': min_samples_leaf,
        "show": XGBClassifier
    },
    # Linear models
    "LogisticRegression": {
        "import": "sklearn.linear_model",
        "penalty": penalty,
        "C": penalty_factor,
        "dual": [False],
        "show": LogisticRegression
    },
    # Naive_bayes
    "BernoulliNB": {
        "import": "sklearn.naive_bayes",
        "alpha": penalty_factor,
        "fit_prior": dual,
        "show": BernoulliNB
    },
    "GaussianNB": {
        "import": "sklearn.naive_bayes",
        "show": GaussianNB
    },
    # Neighbors
    "KNeighborsClassifier": {
        "import": "sklearn.neighbors",
        "weights": weights,
        "p": [1, 2],
        "n_neighbors": [50],
        "show": KNeighborsClassifier
    }
}
