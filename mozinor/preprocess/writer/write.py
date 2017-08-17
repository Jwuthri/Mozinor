# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import pandas as pd


def writeDataframe(X, y, filepath):
    df = pd.concat([X, y], axis=1)
    filename = filepath.split("/")[-1]
    filepath = "Process" + filename
    df.to_csv(filepath, index=False)

    return filepath
