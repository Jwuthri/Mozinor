# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging

import pandas as pd

from mozinor.preprocess.settings import logger


class Dummify(object):
    """Module to transform categoric variable into dummies."""

    def __init__(self, dataframe, dummiefy=True):
        """Dummify categoric variables.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                dummiefy (bool): apply dummiefication or not,
                    by default True
        """
        self.dataframe = dataframe
        self.dummiefy = dummiefy

    def dummiefication(self, dataframe):
        """Transform categoric variables into dummiees.

            Args:
            -----
                dataframe (pandas.DataFrame): data

            Return:
            --------
                pandas.DataFrame with new columns based on cataegoric value
        """
        dummify = dataframe.loc[:, dataframe.dtypes == object]
        for col in dummify.columns:
            logger.log("Dumify column {}".format(col), logging.DEBUG)
            if len(dataframe[col].unique()) < 10:
                df = pd.get_dummies(dataframe[col], drop_first=True, prefix=col)
                dataframe = pd.concat([df, dataframe], axis=1)

        return dataframe

    def dummies(self):
        """Clean a dataframe.

            Return:
            -------
                pandas.DataFrame with dummies
        """
        dataframe = self.dataframe.copy()
        if self.dummiefy:
            dataframe = self.dummiefication(dataframe)

        return dataframe
