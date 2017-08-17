# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging
import itertools

import numpy as np

from mozinor.preprocess.settings import logger


class FeatureEngineering(object):
    """This module provide some tools to make some transformations
    on feature, to add some news."""

    def __init__(self, dataframe, cols=None,
                 transformations=["log", "sqrt", "square"], derivate=True):
        """Make some feature engineering on the data.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                cols (list): list of columns to work on
                transformations (list): all transformatio applied,
                    if None no transformation are made
                    by default (["log", "sqrt", "square"])
                derivate (bool): made the combinaison of all possible pairs
                    by default True
        """
        self.dataframe = dataframe
        self.transformations = transformations
        self.derivate = derivate
        self.cols = self.set_cols(cols)
        self.cols_transform = self.colsToTransform()

    def set_cols(self, cols):
        """Set the cols to work on.

            Args:
            -----
                cols (list): columns to work on

            Return:
            -------
                list of all columns to work on
        """
        if cols:
            return cols
        else:
            return list(self.dataframe.columns)

    def colsToTransform(self):
        """Columns on which we can apply a transformation.

            Returns:
            --------
                list of columns which are not categoric
        """
        return list(self.dataframe[self.cols].select_dtypes(
            include=["float", "float64", "int", "int64"]
        ).columns)

    def transformCol(self, col):
        """Transform one column

            Args:
            -----
                col (str): columns to transform

            Return:
            -------
                pandas.DataFrame with the new columns based on the transformations
        """
        for trans in self.transformations:
            self.dataframe[trans + col] = self.dataframe[col].apply(getattr(np, trans))

    def transformCols(self):
        """Transform all columns from colsToTransform.columns.
        Complexity 3 * n (n = nb_cols).

            Return:
            -------
                pandas.DataFrame with the new columns
        """
        for col in self.cols_transform:
            logger.log("Transform {}".format(col), logging.DEBUG)
            self.transformCol(col)

    def derivateCol(self, col1, col2):
        """Derive the column 1 with the column 2.

            Args:
            -----
                col1 (str): numerator
                col2 (str): denominator

            Return:
            -------
                Add new column based on the derivate value
        """
        derivate = self.dataframe[col1] / self.dataframe[col2]
        self.dataframe["derivate_" + col1 + "_div_" + col2] = derivate
        derivate = self.dataframe[col1] * self.dataframe[col2]
        self.dataframe["derivate_" + col1 + "_mult_" + col2] = derivate

    def derivateCols(self):
        """Derivate all the columns.
        Complexity n * n (n = nb_cols)

            Return:
            -------
                pandas.DataFrame with all new columns derivate
        """
        possible_derivatif = itertools.product(self.cols_transform, self.cols_transform)
        for col1, col2 in possible_derivatif:
            # Trash hack find a best way, pls
            if col1 != col2:
                logger.log("Derivate {} / {}".format(col1, col2), logging.DEBUG)
                self.derivateCol(col1, col2)

    def featurize(self):
        """Build new features.

            Return:
            -------
                pandas.DataFrame with all the new features
        """
        if self.transformations:
            self.transformCols()
        else:
            logger.log("We won't transform features", logging.WARNING)
        if self.derivate:
            self.derivateCols()
        else:
            logger.log("We won't derivate features", logging.WARNING)

        return self.dataframe
