# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging
import datetime
import pydotplus
from io import StringIO

import numpy as np
import pandas as pd

from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mozinor.config.classifiers import *
from mozinor.config.regressors import *
from mozinor.config.params import *
from mozinor.config.explain import *
from mozinor.settings import logger


class EvaluateModels(object):

    def __init__(self, X_train, y_train, X_test, y_test, is_regression, fast=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_regression = is_regression
        self.fast = fast
        self.setPipeline()
        self.setOptimalNeighbors()
        self.setOptimalTrees()

    def setPipeline(self):
        if self.is_regression:
            if self.fast:
                self.pipeline = Fast_Regressors
            else:
                self.pipeline = Regressors
        else:
            if self.fast:
                self.pipeline = Fast_Classifiers
            else:
                self.pipeline = Classifiers

    def setOptimalNeighbors(self):
        logger.log("Optimal number of clusters", logging.INFO)
        print(Elbow)
        self.optimal_neighbors = OptimalNeighbors(self.X_train).compute()

    def setOptimalTrees(self):
        logger.log("Optimal number of trees", logging.INFO)
        print(OobError)
        self.optimal_trees = OptimalTrees(
            self.X_train, self.y_train, self.is_regression
        ).compute()

    def instanciateEstimator(self, estimator, params):
        query = 'from {} import {}'
        cquery = query.format(params.get("import"), estimator)
        exec(cquery)

        return eval(estimator)

    def updateDict(self, dictionary, key):
        dcopy = dict(dictionary)
        del dcopy[key]

        return dcopy

    def updateParams(self, estimator, params):
        logger.log("Estimator {}".format(estimator), logging.INFO)
        params = self.updateDict(params, "import")
        if estimator in ["KNeighborsClassifier", "KNeighborsRegressor"]:
            params["n_neighbors"] = [self.optimal_neighbors]

        if "show" in params:
            print(params.get("show"))
            del params["show"]

        return params

    def _getBestParams(self, cv):
        logger.log("   Best params => {}".format(str(cv.best_params_)), logging.INFO)
        logger.log("   Best Score => {0:.3f}".format(abs(cv.best_score_)), logging.INFO)

        return cv

    def getBestParams(self, cv):
        X_train = np.array(self.X_train)
        y_train = np.array(self.y_train.ix[:,0])
        cv.fit(X_train, y_train)

        return self._getBestParams(cv)

    def buildRandomizedSearchCV(self, estimator_cls, params):
        cv = RandomizedSearchCV(
            estimator=estimator_cls(),
            param_distributions=params,
			n_iter=10,
			cv=3,
			verbose=1,
			n_jobs=1
        )

        return self.getBestParams(cv)

    def buildGridSearchCV(self, estimator_cls, params):
        cv = GridSearchCV(
            estimator=estimator_cls(),
            param_grid=params,
			verbose=1,
			n_jobs=1
        )

        return self.getBestParams(cv)

    def runPipe(self, estimator_cls, params):
        try:
            return self.buildRandomizedSearchCV(estimator_cls, params)
        except Exception:
            return self.buildGridSearchCV(estimator_cls, params)

    def showDecisionTree(self, cv):
        dot_data = StringIO()
        export_graphviz(
            cv.best_estimator_, out_file=dot_data,
            feature_names=list(self.X_train.columns),
            filled=True, rounded=True, special_characters=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        filename = str(datetime.datetime.now()).replace(" ", "") + ".png"
        graph.write_png(filename)
        logger.log("Check the decision tree: {}".format(filename), logging.INFO)

    def _makePipe(self, estimator, params):
        estimator_cls = self.instanciateEstimator(estimator, params)
        params = self.updateParams(estimator, params)
        cv = self.runPipe(estimator_cls, params)
        if estimator in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
            self.showDecisionTree(cv)

        return {
            "best_estimator_": cv.best_estimator_,
            "best_score_": abs(cv.best_score_),
            "best_params_": cv.best_params_
        }

    def makePipe(self, estimator, params):
        try:
            return self._makePipe(estimator, params)
        except Exception:
            return {
                "best_estimator_": estimator,
                "best_score_": 0
            }

    def evaluate(self):
        d_model_score = dict()
        for estimator, params in self.pipeline.items():
            d_model_score[estimator] = self.makePipe(estimator, params)

        return d_model_score
