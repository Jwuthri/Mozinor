# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging
from itertools import combinations

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score

from tqdm import tqdm
from vecstack import stacking

from mozinor.pipeline import EvaluateModels
from mozinor.config.params import *
from mozinor.config import explain
from mozinor.settings import logger


class Stacking(object):

    def __init__(self, X_train, y_train, X_test, y_test, is_regression):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_regression = is_regression
        self.setMetric()

    def setMetric(self):
        if self.is_regression:
            self.metric = r2_score
        else:
            self.metric = accuracy_score

    def applyPolynomialFeatures(self, df, degree):
        poly_reg = PolynomialFeatures(degree=degree)

        return pd.DataFrame(poly_reg.fit_transform(df))

    def getModelsScore(self, degree):
        X_train = self.applyPolynomialFeatures(self.X_train, degree)
        X_test = self.applyPolynomialFeatures(self.X_test, degree)

        return EvaluateModels(
            X_train, self.y_train, X_test, self.y_test, self.is_regression
        ).evaluate()

    def explainPolynomialFeatures(self, degree):
        logger.log("Work on PolynomialFeatures: degree {}".format(str(degree)), logging.INFO)
        print(explain.PolynomialFeatures)

    def getModelsScoreDegree(self):
        d_summary_models_degree = dict()
        for degree in ldegrees:
            self.explainPolynomialFeatures(degree)
            d_summary_models_degree[degree] = self.getModelsScore(degree)

        return d_summary_models_degree

    def extractModelScore(self, d_model_score, degree):
        l2, l1 = [], []
        for estimator, model in d_model_score.items():
            l1.append(model.get("best_score_"))
            l2.append(model.get("best_estimator_"))
        df = pd.DataFrame(data={"Score": l1, "Estimator": l2})
        df["Degree"] = degree

        return df

    def dict2frame(self, d_models_score_degree):
        df = pd.DataFrame()
        for degree, d_model_score in d_models_score_degree.items():
            temp_df = self.extractModelScore(d_model_score, degree)
            df = pd.concat([df, temp_df])
        df = df.sort_values(by=["Score"], ascending=False).reset_index(drop=True)
        logger.log("{}".format(df), logging.INFO)

        return df

    def whichDegreeToWorkOn(self, df):
        df["SumScore"] = df.groupby("Degree")["Score"].transform("sum")
        df = df.sort_values(by="SumScore", ascending=False)

        return df.iloc[0]["Degree"]

    def pandas2numpy(self, X_train, y_train, X_test, y_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train.ix[:,0])
        X_test = np.array(X_test)
        y_test = np.array(y_test.ix[:,0])

        return X_train, y_train, X_test, y_test

    def bestModelsDegree(self, df_models_score_degree, degree):
        sub_df = df_models_score_degree[df_models_score_degree["Degree"] == degree]

        return list(sub_df["Estimator"])

    def naiveStacking(self, degree, lmodels):
        print(explain.Stacking)
        X_train = self.applyPolynomialFeatures(self.X_train, degree)
        X_test = self.applyPolynomialFeatures(self.X_test, degree)
        X_train, y_train, X_test, y_test = self.pandas2numpy(
            X_train, self.y_train, X_test, self.y_test)

        return stacking(
            lmodels, X_train, y_train, X_test,
            regression=self.is_regression, metric=self.metric,
            n_folds=3, shuffle=True, random_state=0, verbose=1
        ), y_train, y_test

    def possibleCombination(self, lmodels):
        combi = list(range(len(lmodels)))
        output = [map(list, combinations(combi, i)) for i in range(1, len(combi) + 1)]
        flatten = lambda l: [item for output in l for item in output]

        return flatten(output)

    def stackedModels(self, combination, lmodels):
        models = list()
        for i in combination:
            models.append(lmodels[i])

        return models

    def firstLevelModel(self, S_train, S_test, combination, lmodels):
        train = S_train[:, combination]
        test = S_test[:, combination]
        S_models = self.stackedModels(combination, lmodels)

        return train, test, S_models

    def secondLevelModel(self, X_train, y_train, X_test, y_test, lmodels):
        best_score = 0
        best_model = None
        for model in lmodels:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = self.metric(y_test, y_pred)
            if score > best_score:
                best_score = score
                best_model = model

        return model, score

    def bestCombinationModels(self, S_train, S_test, y_train, y_test, lpossible_combination, lmodels):
        dict_stack_models = dict()
        pbar = tqdm(total=len(lpossible_combination))
        for combination in lpossible_combination:
            n_stack = len(combination)
            pbar.set_description("Stacking %s models" % n_stack)
            X_train, X_test, S_models = self.firstLevelModel(S_train, S_test, combination, lmodels)
            model, score = self.secondLevelModel(X_train, y_train, X_test, y_test, lmodels)
            dict_stack_models[score] = {
                "Fit1stLevelEstimator": S_models,
                "Fit2ndLevelEstimator": model,
            }
            pbar.update(1)
        pbar.close()

        return dict_stack_models

    def setBestModel(self, df_models_score_degree):
        self.best_model = df_models_score_degree.iloc[0]

    def setBestStackModels(self, dict_stack_models, degree):
        sorted_dict_stack_models = sorted(dict_stack_models)
        best_score = sorted_dict_stack_models[-1]
        best_stack_models = dict_stack_models.get(best_score)
        df_best_stack_models = pd.DataFrame.from_dict([best_stack_models])
        df_best_stack_models["Score"] = best_score
        df_best_stack_models["Degree"] = degree
        self.best_stack_models = df_best_stack_models.iloc[0]

    def run(self):
        d_models_score_degree = self.getModelsScoreDegree()
        df_models_score_degree = self.dict2frame(d_models_score_degree)
        self.setBestModel(df_models_score_degree)
        degree = self.whichDegreeToWorkOn(df_models_score_degree)
        lmodels = self.bestModelsDegree(df_models_score_degree, degree)
        (S_train, S_test), y_train, y_test = self.naiveStacking(degree, lmodels)
        lpossible_combination = self.possibleCombination(lmodels)
        dict_stack_models = self.bestCombinationModels(
            S_train, S_test, y_train, y_test, lpossible_combination, lmodels)
        self.setBestStackModels(dict_stack_models, degree)
