# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


def optimalPcaComponents(pca_ratio):
    """Compute the optimal number of components for the pca.

        Arg:
        ----
            pca_ration (pca.explained_variance_ratio_): var explained

        Return:
        -------
            int, of optimal pc components
    """
    for i in range(2, len(pca_ratio)):
        pca_explication = sum(pca_ratio[:i])
        if pca_explication > 0.99:
            return i
    return i


def plotPca(pca_ratio):
    """Plot the pca curve.

        Arg:
        ----
            pca_ration (pca.explained_variance_ratio_): var explained
    """
    plt.bar(np.arange(len(pca_ratio)) + 1, pca_ratio)
    plt.title("Variance expliquée")
    plt.show()


def buildPca(df, n_components=100):
    """Make a pca decomposition.

        Args:
        -----
            df (pandas.DataFrame): datas
            n_components (int): nb of components

        Return:
        -------
            pca model
    """
    return PCA(n_components=n_components).fit(df)


def nbPca(df):
    """Define the number of components.

        Arg:
        ----
            df (pandas.DataFrame): datas

        Return:
        -------
            int, of optimal number of components
    """
    pca = buildPca(df)
    pca_ratio = pca.explained_variance_ratio_
    plotPca(pca_ratio)

    return optimalPcaComponents(pca_ratio)


def applyPca(df):
    """Build a dataframe based on PCA decomposition.

        Arg:
        ----
            df (pandas.DataFrame): datas

        Return:
        -------
            pandas.DataFrame with PCA columns
    """
    nb_pca = nbPca(df)
    pca = buildPca(df, n_components=100)
    df = pd.DataFrame(pca.transform(df))
    for col in df.columns:
        df.rename(columns={col: "PCA_" + str(col)}, inplace=True)

    return df
