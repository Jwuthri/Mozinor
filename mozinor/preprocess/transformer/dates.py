# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import pandas as pd


def isDateTime(row):
    try:
        row.hour
        return True
    except Exception:
        return False


def colDateType(df):
    date = []
    date_time = []

    for col in df.columns:
        row = df[col].iloc[0]
        try:
            row.day
            if isDateTime(row):
                date_time.append(col)
            else:
                date.append(col)
        except Exception:
            pass

    return date, date_time


def frameFromDatetimeCol(serie, col):
    moment = serie.apply(lambda x: 0 if x.hour < 10 else 1 if 10 <= x.hour < 18 else 2)

    day = serie.apply(lambda x: x.day)
    month = serie.apply(lambda x: x.month)
    year = serie.apply(lambda x: x.year)

    weekday = serie.apply(lambda x: 0 if x.weekday() > 4 else 1)

    data = dict()
    for arg in ["moment", "day", "month", "year", "weekday"]:
        data[col + "_" + arg] = vars()[arg]

    return pd.DataFrame(data)


def frameFromDateCol(serie, col):
    day = serie.apply(lambda x: x.day)
    month = serie.apply(lambda x: x.month)
    year = serie.apply(lambda x: x.year)

    weekday = serie.apply(lambda x: 0 if x.weekday() > 4 else 1)

    data = dict()
    for arg in ["day", "month", "year", "weekday"]:
        data[col + "_" + arg] = vars()[arg]

    return pd.DataFrame(data)


def buildColsFromDateCols(df):
    date, date_time = colDateType(df)
    for col in date_time:
        new_df = frameFromDatetimeCol(df[col], col)
        df = pd.concat([df, new_df], axis=1)

    for col in date:
        new_df = frameFromDateCol(df[col], col)
        df = pd.concat([df, new_df], axis=1)

    return df
