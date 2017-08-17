# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging

import pandas as pd

from mozinor.preprocess.settings import logger


def readFile(filepath, encoding="utf-8-sig", sep=",", infer_datetime=True, decimal=',', thousands='.'):
    """Read a csv file.

        Args:
        -----
            filepath (str): the path of the data file
            encoding (str): the encoding type
            sep (char): the delimiter
            infer_datetime (bool): try to optimaze datetime

        Return:
        -------
            pandas.DataFrame with data
        """

    def getColumns(dargs):
        """Get all columns names.

            Arg:
            -----
                dargs (dict): args to read the csv file

            Return:
            -------
                list of all columns in the dataframe
        """
        dargs.update({"nrows": 5})

        return list(pd.read_csv(**dargs).columns)

    dargs = {
        "encoding": encoding, "sep": sep, "decimal": decimal,
        "engine": "python", "filepath_or_buffer": filepath, "thousands": thousands
    }
    logger.log("Read csv file: {}".format(filepath), logging.DEBUG)
    columns = getColumns(dargs)
    del dargs["nrows"]

    if infer_datetime:
        dargs.update({
            "parse_dates": columns, "infer_datetime_format": infer_datetime
        })
        logger.log("args: {}".format(str(dargs)), logging.DEBUG)

    return pd.read_csv(**dargs)


def GetX_Y(df, col_y, col_to_remove=[]):
    """Select X and y dataframes.

        Args:
        -----
            df (pandas.DataFrame): the datas
            col_y (str): col to predict
            col_to_remove (list): columns you don't want to use

        Returns:
        -------
            pandas.DataFrame X and y
    """
    y = df[[col_y]]
    X = df.drop([col_y], axis=1)
    for col in col_to_remove:
        if col in X:
            X.drop([col], axis=1, inplace=True)
            logger.log("Remove column {}".format(col), logging.DEBUG)
        else:
            logger.log("Col {} not in the dataframe".format(col), logging.WARNING)

    return X, y
