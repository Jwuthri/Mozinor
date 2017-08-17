# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import warnings
import logging

from sklearn.preprocessing import StandardScaler

from mozinor.preprocess.settings import logger


class ScaleData(object):
    """Module to scale feature with big entropy."""

    def __init__(self, dataframe, scaled=True):
        """Scale the dataframe.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                scale (bool): scale the data ?
                    by default True
        """
        self.dataframe = dataframe
        self.scaled = scaled

    def scaleCol(self, serie, col):
        """Scale the serie.

            Args:
            -----
                serie (pandas.Serie): serie to scale

            Return:
            -------
                pandas.Serie scaled
        """
        warnings.filterwarnings("ignore")
        try:
            return StandardScaler().fit_transform(serie)
        except Exception as e:
            logger.log("{}".format(e), logging.ERROR)
            return serie

    def scaleCols(self, dataframe):
        """Determine features to scale.

            Args:
            -----
                dataframe (pandas.DataFrame): data

            Return:
            -------
                pandas.DataFrame with columns scaled
        """
        for col in dataframe.columns:
            logger.log("Scale column {}".format(col), logging.DEBUG)
            dataframe[col] = self.scaleCol(dataframe[col], col)

        return dataframe

    def scale(self):
        """Scale the features.

            Return:
            -------
                pandas.DataFrame with features (with big entropy) scaled
        """
        dataframe = self.dataframe.copy()
        if self.scaled:
            dataframe = self.scaleCols(dataframe)

        return dataframe
