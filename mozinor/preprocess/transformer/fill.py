# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from mozinor.preprocess.settings import logger


class FillNaN(object):
    """Module to fill NaN thx to a clustering approach."""

    def __init__(self, dataframe):
        """Fill NaN with a clustering.

            Args:
            -----
                dataframe (pandas.DataFrame): data
        """
        self.dataframe = dataframe
        self.cols = self.noCategoricCols()

    def noCategoricCols(self):
        """Select only non categoric cols.

            Return:
            -------
                List of columns no categoric
        """
        return list(self.dataframe.select_dtypes(
            include=["float", "float64", "int", "int64"]
        ).columns)

    def wcss(self, dataframe, max_cluster=20, plot=True):
        """Determine the best number of cluster.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                max_cluster (int): the max number of cluster possible
                plot (bool): print the plot

            Return:
            -------
                list of wcss values
        """
        dataframe = dataframe.fillna(dataframe.mean())
        wcss = list()
        for i in range(1, max_cluster):
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(dataframe[self.cols])
            wcss.append(kmeans.inertia_)
        if plot:
            plt.plot(range(1, max_cluster), wcss)
            plt.title('Elbow')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()

        return wcss

    def computeLcurve(self, wcss):
        """Compute the lcurve.

            Args:
            -----
                wcss (list): the elbow values

            Return:
            -------
                dict k/v: number of clusters / lcurve value
        """
        d_derivate = dict()
        for i in range(1, len(wcss) - 1):
            d_derivate[i] = wcss[i + 1] + wcss[i - 1] - 2 * wcss[i]

        return d_derivate

    def bestLcurveValue(self, d_derivate):
        """Select best lcurve value, the one that doesn't decrease anymore.

            Args:
            -----
                d_derivate (dict): dict of nb_cluster / lcurve

            Return:
            -------
                int, value of the optimal nb cluster
        """
        nb_cluster = len(d_derivate)
        for k, v in d_derivate.items():
            if v < 0:
                return k

        return nb_cluster

    def computeOptimalCluster(self, wcss):
        """Select the optimal number of clusters.

            Args:
            -----
                wcss (list): list of wcs values

            Return:
            -------
                int, of the optimal number of clusters
        """
        d_derivate = self.computeLcurve(wcss)

        return self.bestLcurveValue(d_derivate)

    def nbCluster(self):
        """Choose the number of cluster thanks to Elbow method.

            Return:
            -------
                Number of cluster
        """
        user_input = input('How many clusters do you want ? ')
        try:
            return int(user_input)
        except ValueError:
            raise ValueError("An int is requiered")

    def clustering(self, dataframe, nb_cluster=2):
        """Make a knn based on all columns.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                nb_cluster (int): number of cluster, determined by elbow

            Return:
            -------
                pandas.Serie contains the cluster for each rows
        """
        dataframe = dataframe.fillna(dataframe.mean())
        kmeans = KMeans(n_clusters=nb_cluster, init='k-means++')

        return kmeans.fit_predict(dataframe[self.cols])

    def meanCluster(self, dataframe, col):
        """Take the mean for each cluster/col.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                col (str): the column to work on

            Return:
            -------
                dict contains the k/v for each cluster and mean value
        """
        d_cluster = dict()
        for cluster in dataframe["Cluster"].unique():
            d_cluster[cluster] = dataframe[
                dataframe["Cluster"] == cluster
            ][col].mean()

        return d_cluster

    def fillCol(self, dataframe, col):
        """Fill NaN of a column thanks to dict of cluster values.

            Args:
            -----
                dataframe (pandas.DataFrame): data
                col (str): column to work on

            Return:
            -------
                pandas.Serie with NaN filled
        """
        d_cluster = self.meanCluster(dataframe, col)
        nan_serie = dataframe[~dataframe[col].notnull()]
        for idx, row in nan_serie.iterrows():
            value = d_cluster.get(row["Cluster"])
            dataframe.set_value(idx, col, value)

        return dataframe[col]

    def fillCols(self, dataframe):
        """Fill NaN for all columns.

            Args:
            -----
                dataframe (pandas.DataFrame): data

            Return:
            -------
                pandas.DataFrame with new value instead of NaN
        """
        for col in self.cols:
            logger.log("Filling NaN, column: {}".format(col), logging.DEBUG)
            dataframe[col] = self.fillCol(dataframe, col)

        return dataframe.drop("Cluster", axis=1)

    def fill(self):
        """Fill the dataframe.

            Return:
            -------
                pandas.DataFrame filled
        """
        dataframe = self.dataframe.copy()
        wcss = self.wcss(dataframe)
        nb_cluster = self.computeOptimalCluster(wcss)
        logger.log("Optimal nb of cluster is: {}".format(nb_cluster), logging.DEBUG)
        dataframe["Cluster"] = self.clustering(dataframe, nb_cluster)

        return self.fillCols(dataframe)
