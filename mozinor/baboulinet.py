# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from mozinor.script_generator import solo_model_code, stack_model_code
from mozinor.preprocess.processing import Preprocess
from mozinor.settings import logger
from mozinor.stack import Stacking


class Baboulinet(Preprocess):

    def __init__(self, filepath, y_col, regression, process=False, sep=',',
                 col_to_drop=[], derivate=False, transform=False, scaled=True,
                 infer_datetime=True, encoding="utf-8-sig", dummify=True):
        super().__init__(
            filepath, y_col, sep, col_to_drop, derivate, transform,
            infer_datetime, encoding, dummify, scaled)
        self.regression = regression
        if process:
            logger.log("Process the data", logging.INFO)
            self.filepath = self.process()
        else:
            self.filepath = filepath

    def babouline(self):
        df = self.read()
        y = df[[self.y_col]]
        X = df.drop(self.y_col, axis=1)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        self.stc = Stacking(x_train, y_train, x_test, y_test, self.regression)
        self.stc.run()

        return self.stc

    def bestModelScript(self):
        dscript = {
            "filepath": self.filepath,
            "col_to_pred": self.y_col,
            "regression": self.regression,
            "degree": self.stc.best_model["Degree"],
            "model": self.stc.best_model["Estimator"],
            "module": self.stc.best_model["Estimator"].__module__,
            "name": self.stc.best_model["Estimator"].__class__.__name__
        }
        script = solo_model_code.format(**dscript)
        script = script.split("\n")
        path = self.filepath.split(".")[0] + "_solo_model_script.py"
        with open(path, "w") as text_file:
            for row in script:
                print(row,  file=text_file)
        logger.log("Check script file {}".format(path), logging.INFO)

        return path

    def bestStackModelScript(self):
        dscript = {
            "filepath": self.filepath,
            "col_to_pred": self.y_col,
            "regression": self.regression,
            "degree": self.stc.best_stack_models["Degree"],
            "models_lvl1": self.stc.best_stack_models["Fit1stLevelEstimator"],
            "model_lvl2": self.stc.best_stack_models["Fit2ndLevelEstimator"],
        }
        script = stack_model_code.format(**dscript)
        script = script.split("\n")
        path = self.filepath.split(".")[0] + "_stack_model_script.py"
        with open(path, "w") as text_file:
            for row in script:
                print(row,  file=text_file)
        logger.log("Check script file {}".format(path), logging.INFO)

        return path
