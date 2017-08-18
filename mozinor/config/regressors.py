# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
from mozinor.config.params import *
from mozinor.config.explain import *


Fast_Regressors = {
    "DecisionTreeRegressor": {
        "import": "sklearn.tree",
        "criterion": ["mse", "mae"],
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": DecisionTreeRegressor
    },
    "ExtraTreesRegressor": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "max_features": max_features,
        "bootstrap": dual,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": ExtraTreesRegressor
    },
    "ElasticNetCV": {
        "import": "sklearn.linear_model",
        "l1_ratio": max_features,
        "tol": learning_rate,
        "show": ElasticNetCV
    },
    "LassoLarsCV": {
        "import": "sklearn.linear_model",
        "normalize": dual,
        "show": LassoLarsCV
    },
    "RidgeCV": {
        "import": "sklearn.linear_model",
        "show": RidgeCV
    },
    'XGBRegressor': {
        "import": "xgboost",
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': max_features,
        'min_child_weight': min_samples_leaf,
        "show": XGBRegressor
    }
}

Regressors = {
    # Tree
    "DecisionTreeRegressor": {
        "import": "sklearn.tree",
        "criterion": ["mse", "mae"],
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": DecisionTreeRegressor
    },
    # Ensemble
    "ExtraTreesRegressor": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "max_features": max_features,
        "bootstrap": dual,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": ExtraTreesRegressor
    },
    "RandomForestRegressor": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "max_features": max_features,
        "bootstrap": dual,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": RandomForestRegressor
    },
    "GradientBoostingRegressor": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "learning_rate": learning_rate,
        "max_features": max_features,
        "loss": gbloss,
        "alpha": alpha,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "show": GradientBoostingRegressor
    },
    "AdaBoostRegressor": {
        "import": "sklearn.ensemble",
        'n_estimators': n_estimators,
        "learning_rate": learning_rate,
        "loss": adaloss,
        "show": AdaBoostRegressor
    },
    # XGBoost
    'XGBRegressor': {
        "import": "xgboost",
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': max_features,
        'min_child_weight': min_samples_leaf,
        "show": XGBRegressor
    },
    # Linear models
    "ElasticNetCV": {
        "import": "sklearn.linear_model",
        "l1_ratio": max_features,
        "tol": learning_rate,
        "show": ElasticNetCV
    },
    "LassoLarsCV": {
        "import": "sklearn.linear_model",
        "normalize": dual,
        "show": LassoLarsCV
    },
    "RidgeCV": {
        "import": "sklearn.linear_model",
        "show": RidgeCV
    },
    # Neighbors
    "KNeighborsRegressor": {
        "import": "sklearn.neighbors",
        "weights": weights,
        "p": [1, 2],
        "n_neighbors": [50],
        "show": KNeighborsRegressor
    }
}
