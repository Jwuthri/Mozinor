# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import string
import logging

from mozinor.preprocess.settings import logger


def formatStr(bad_str, hard=True):
    """Remove trash punctuation.

        Args:
        -----
            bad_str (str): string to format
            hard (bool): remove all punctuation or just remove space

        Return:
        -------
            string formated
    """
    try:
        if hard:
            for char in string.punctuation + ' ':
                bad_str = bad_str.replace(char, '')

        return " ".join(bad_str.split()).lower()
    except Exception:
        # logger.log("Can't format {}".format(bad_str), logging.WARNING)
        return bad_str


def formatCols(df):
    """Formats all object columns of the dataframe.

        Arg:
        ----
            df (pandas.DataFrame): datas

        Return:
        -------
            pandas.DataFrame formatted
    """
    cols = df.select_dtypes(include=["object"]).columns
    for col in cols:
        logger.log("Format col: {}".format(col), logging.DEBUG)
        df[col] = df[col].map(formatStr)
        try:
            res = df[col].astype(float)
            logger.log("Col {} has been cast into float".format(col), logging.DEBUG)
        except Exception:
            res = df[col]
        finally:
            df[col] = res

    return df
