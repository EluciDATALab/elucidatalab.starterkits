# ©, 2019, Sirris
# owner: HCAB

"""Wrapper implementation of the kmedoids algorithm around pyclustering"""
import numpy as np
from pyclustering.cluster import kmedoids as py_kmedoids

from sklearn import metrics as sk_metrics
from scipy.stats import rankdata

from starterkits.utils.manager import progressbar


class KMedoids:
    """
    Wrapper implementation of the kmedoids algorithm

    Based on package pyclustering, it allows for the iteration of multiple
    clustering runs and selection of the run with highest clustering quality

    E.g.: KMedoids(4).fit(X)

    Relevant attributes
    ----------
    best_cluster_labels: array.
        Contains the cluster labels of each sample from the iteration that
        obtained the highest cluster quality score

    best_cluster_quality: array.
        Contains the quality (for each of the considered metrics) of the best
        iteration
    """

    def __init__(self,
                 number_of_clusters,
                 number_of_iterations=10,
                 metrics=None,
                 verbose=False,
                 initial_medoids=None):
        """
        Kmedoids implementation allowing for multiple iterations

        :param number_of_clusters: int. number of clusters
        :param number_of_iterations: int. number of iterations. Default: 10
        :param metrics: list/str. List of quality metrics to be used when
         evaluating cluster quality. Must be one from module sklearn.metrics.
         Default: ['davies_bouldin_score', 'calinski_harabasz_score']
        :param verbose: bool. Whether to print progress. Default: False
        :param initial_medoids: list/array. List of initial medoids.
         Default: None

        """
        if metrics is None:
            metrics = ['davies_bouldin_score', 'calinski_harabasz_score']

        self.number_of_clusters = number_of_clusters
        self.number_of_iterations = number_of_iterations
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.verbose = verbose
        self.initial_medoids = initial_medoids
        self.metric_functions_mapping = {
            metric: getattr(sk_metrics, metric)
            for metric in self.metrics}
        self.X = None
        self.cluster_labels = None
        self.quality = None
        self.best_cluster_labels = None
        self.best_cluster_quality = None

    def fit(self, data, kwargs=None):
        """
        Fit kmedoids algorithm

        :param data: array. Array to be clustered in the form
         observations/features
        :param kwargs: dictionary. Further parameters to py_kmedoids.kmedoids.
         Default: None
        """
        kwargs = {} if kwargs is None else kwargs
        quality = np.empty((len(self.metrics), self.number_of_iterations))
        cluster_labels = {}
        for i in progressbar(np.arange(self.number_of_iterations),
                             verbal=self.verbose):
            initial_medoids = (np.random.choice(data.shape[1],
                                                self.number_of_clusters)
                               if self.initial_medoids is None
                               else self.initial_medoids)
            clusters = (py_kmedoids.kmedoids(data, initial_medoids, **kwargs)
                        .process()
                        .get_clusters())
            current_cluster_labels = self._get_cluster_labels(clusters,
                                                              data.shape[0])
            cluster_labels[i] = current_cluster_labels
            quality[:, i] = self._get_partition_quality(data,
                                                        current_cluster_labels)

        self.X = data
        self.cluster_labels = cluster_labels
        self.quality = quality

        best = self._get_best_iteration()
        self.best_cluster_labels = self.cluster_labels[best]
        self.best_cluster_quality = self.quality[:, best]

    @staticmethod
    def _get_cluster_labels(clusters, number_of_rows):
        cluster_labels = np.empty(number_of_rows)
        for cluster_label, cluster_index in enumerate(clusters):
            cluster_labels[cluster_index] = cluster_label
        return cluster_labels

    def _get_partition_quality(self, data, cluster_labels):
        quality = [self.metric_functions_mapping[metric](data, cluster_labels)
                   for metric in self.metrics]
        if 'calinski_harabasz_score' in self.metrics:
            index_score = [metric_index
                           for metric_index, metric in enumerate(self.metrics)
                           if metric == 'calinski_harabasz_score'][0]
            quality[index_score] = quality[index_score] * -1

        return quality

    def _get_best_iteration(self):
        ranked_quality = np.apply_along_axis(rankdata, 1, self.quality)
        ranked_quality = np.mean(ranked_quality, 0)

        return np.argmin(ranked_quality)
