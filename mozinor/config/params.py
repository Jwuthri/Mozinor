# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
from collections import OrderedDict
import matplotlib.pyplot as plt
import warnings
import math

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


alpha = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
adaloss = ["linear", "square", "exponential"]
criterion = ["gini", "entropy"]
dual = [True, False]
gbloss = ["ls", "lad", "huber", "quantile"]
kernel = ["linear", "poly", "rbf", "sigmoid"]
ldegrees = [1, 2]
learning_rate = [0.01, 0.1, 0.5, 1.]
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n_estimators = [10, 50, 75, 100]
penalty = ["l1", "l2"]
penalty_factor = [0.01, 0.1, 0.5, 1., 5., 10., 100.]
svcloss = ["hinge", "squared_hinge"]
svrloss = ["epsilon_insensitive", "squared_epsilon_insensitive"]
weights = ["uniform", "distance"]


class OptimalNeighbors(object):
    """Module to define the optimal number of neighbors."""

    def __init__(self, X_train):
        self.X_train = X_train

    def wcss(self, plot=True):
        wcss = list()
        for i in range(1, 20):
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(self.X_train)
            wcss.append(kmeans.inertia_)
        if plot:
            plt.plot(range(1, 20), wcss)
            plt.title('Elbow')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()

        return wcss

    def computeLcurve(self):
        wcss = self.wcss()
        d_derivate = dict()
        for i in range(1, len(wcss) - 1):
            d_derivate[i] = wcss[i + 1] + wcss[i - 1] - 2 * wcss[i]

        return d_derivate

    def bestLcurveValue(self, d_derivate):
        nb_cluster = len(d_derivate)
        for k, v in d_derivate.items():
            if v < 0:
                return k

        return nb_cluster

    def compute(self):
        d_derivate = self.computeLcurve()

        return self.bestLcurveValue(d_derivate)


class OptimalTrees(object):
    """Module to check the optimal number of trees."""

    def __init__(self, X_train, y_train, is_regression):
        self.X_train = X_train
        self.y_train = y_train
        self.is_regression = is_regression
        self.error_rate = dict()
        self.setEstimator()

    def setEstimator(self):
        if self.is_regression:
            self.est = RandomForestRegressor
        else:
            self.est = RandomForestClassifier

    def possibleEstimator(self):
        d_possible_rf = dict()
        for feature in ["log2", "sqrt", None]:
            key = "RandomForest => max_features={}".format(feature)
            d_possible_rf[key] = self.est(oob_score=True, max_features=feature)

        return d_possible_rf

    def optimalEstimator(self, estimator, estimator_feature):
        errors = list()
        for nb_tree in range(10, 100, 10):
            estimator.set_params(n_estimators=nb_tree)
            estimator.fit(self.X_train, self.y_train)
            oob_error = 1 - estimator.oob_score_
            self.error_rate[estimator_feature].append((nb_tree, oob_error))
            errors.append(oob_error)

        return errors

    def errorRate(self):
        error_df = pd.DataFrame()
        self.error_rate = OrderedDict(
            (key, []) for key in self.possible_estimator.keys())
        for estimator_feature, estimator in self.possible_estimator.items():
            error_df[estimator_feature] = self.optimalEstimator(estimator, estimator_feature)

        return error_df

    def plotErrorRate(self):
        for label, clf_err in self.error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(1, 100)
        plt.title('OOB')
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()

    def applyLogLoss(self, row):
        return math.log(row["NbTree"]) * row["OptimalTrees"]

    def selectMinScore(self, error_df):
        error_df = error_df.sort_values(by=["Optimal", "NbTree"])

        return error_df.iloc[-1]["NbTree"]

    def bestOOB(self, error_df):
        opti = list()
        error_df["OptimalTrees"] = error_df.sum(axis=1)
        error_df["NbTree"] = error_df.index + 2
        for _, row in error_df.iterrows():
            opti.append(self.applyLogLoss(row))
        error_df["Optimal"] = opti

        return self.selectMinScore(error_df)

    def compute(self):
        self.possible_estimator = self.possibleEstimator()
        error_df = self.errorRate()
        self.plotErrorRate()
        optimal_tree = self.bestOOB(error_df)

        return optimal_tree
