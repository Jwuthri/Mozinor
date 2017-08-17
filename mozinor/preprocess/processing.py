# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging

from mozinor.preprocess.settings import logger
from mozinor.preprocess.reader.read import readFile, GetX_Y
from mozinor.preprocess.transformer.regex import formatCols
from mozinor.preprocess.transformer.fill import FillNaN
from mozinor.preprocess.transformer.dummy import Dummify
from mozinor.preprocess.transformer.dates import buildColsFromDateCols
from mozinor.preprocess.transformer.feature import FeatureEngineering
from mozinor.preprocess.transformer.scale import ScaleData
from mozinor.preprocess.writer.write import writeDataframe


class Preprocess(object):
    """This module apply basic cleaning on csv file,
    and return pandas.DataFrame."""

    def __init__(self,
                 filepath, y_col, sep=',', col_to_drop=[], derivate=True, transform=True,
                 infer_datetime=True, encoding="utf-8-sig", dummify=True, scaled=True):
        """Preprocess csv file.

            Args:
            -----
                filepath (str): path of the file
                y_col (str): column to predict
                sep (char): delimiter
                transformer (bool): polynomial featuring
                derivate (bool): derivate the features
                col_to_drop (list): all columns to remove from the df
                infer_datetime (bool): find automaticly date columns
                encoding (str): type of encoding of your csv file
                dumify (bool): dumify cateforic or not
                scale (bool): scale the data

        """
        self.filepath = filepath
        self.sep = sep
        self.y_col = y_col
        self.derivate = derivate
        self.col_to_drop = col_to_drop
        self.infer_datetime = infer_datetime
        self.encoding = encoding
        self.dummify = dummify
        self.scaled = scaled
        self.transform = transform

    @staticmethod
    def int2Float(df):
        for col in df.columns:
            try:
                df[col].astype(float)
            except Exception:
                pass

        return df

    def read(self):
        logger.log("Reading the file {}".format(self.filepath), logging.INFO)
        df = readFile(
            self.filepath, self.encoding, self.sep, self.infer_datetime)
        print(df.shape)
        logger.log("Inital dtypes is {}".format(df.dtypes), logging.DEBUG)

        return df

    def regex(self, df):
        logger.log("Apply regex on the string rows", logging.INFO)
        df = formatCols(df)
        print(df.shape)
        logger.log("New dtypes is {}".format(df.dtypes), logging.DEBUG)

        return df

    def split(self, df):
        logger.log("Split dataframe, and remove useless cols", logging.INFO)
        X, y = GetX_Y(df, self.y_col, self.col_to_drop)
        print(df.shape)

        return X, y

    def fill(self, df):
        logger.log("Filling the NaN values", logging.INFO)
        df = FillNaN(df).fill()
        print(df.shape)

        return df

    def dummy(self, df):
        logger.log("Dummify categoric variables", logging.INFO)
        df = Dummify(df).dummies()
        print(df.shape)

        return df

    def date_cols(self, df):
        logger.log(
            "Build new columns thanks to the datetime columns", logging.INFO)
        df = buildColsFromDateCols(df)
        print(df.shape)

        return df

    def feature(self, df, orginal_cols):
        logger.log("Make some feature engineering", logging.INFO)
        df = FeatureEngineering(df, cols=orginal_cols, derivate=self.derivate).featurize()
        print(df.shape)

        return df

    def scale(self, df):
        logger.log("Scale the data", logging.INFO)
        df = ScaleData(df).scale()
        df = df.fillna(0)
        print(df.shape)

        return df

    def process(self):
        df = self.read()
        df = self.regex(df)
        df, y = self.split(df)
        df = self.fill(df)
        orginal_cols = list(df.columns)
        if self.dummify:
            df = self.dummy(df)
        df = self.date_cols(df)
        if self.transform:
            df = self.feature(df, orginal_cols)
        if self.scaled:
            df = self.scale(df)
        df = self.int2Float(df)
        filepath = writeDataframe(df, y, self.filepath)

        return filepath
