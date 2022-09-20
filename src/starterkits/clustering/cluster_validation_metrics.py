# ©, 2019, Sirris
# owner: PDAG

"""
A set of scripts computing various metrics to determine the best number of clusters of a dataset.

The following metrics are supported: Silhouette, Calinski-Harabasz, Davies Bouldin, Elbow and connectivity. The
recommended use of the script is to run the function "find_optimal_number_of_clusters()", which will apply the
different metrics at once. We refer to the documentation of that function for more details.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import seaborn as sns


def __apply_one_score(data, max_number_clusters=9, algorithm=KMeans, param=None, metric_to_run=metrics.silhouette_score,
                      metric_name='silhouette_score'):
    """
    Determine the ideal number of clusters using the provided metric.

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of cluster to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)
    :param metric_to_run: the function to use to compute the best number of methods
    :param metric_name: the name of the metric to use (used in the returned dictionary). Default: Silhouette

    :return: a dictionary containing all scores for all the numbers of clusters tested
    """
    if param is None:
        param = {}

    scores = {'number_clusters_tested': [], metric_name: []}
    for number_of_clusters in range(2, max_number_clusters + 1, 1):
        clusterer = algorithm(n_clusters=number_of_clusters).set_params(**param)
        cluster_labels = clusterer.fit_predict(data)
        scores['number_clusters_tested'].append(number_of_clusters)
        if metric_name == 'elbow_score':
            if "inertia_" not in clusterer.__dict__.keys():
                scores[metric_name].append(np.nan)
            else:
                scores[metric_name].append(clusterer.inertia_)
        else:
            scores[metric_name].append(metric_to_run(data, cluster_labels))
    return scores


def find_optimal_number_of_clusters(data,
                                    max_number_clusters=9,
                                    algorithm=KMeans,
                                    param=None,
                                    display=False,
                                    col_wrap=3,
                                    score_metrics=None,
                                    **kwargs):
    """
    Compute various metrics to determine the best number of clusters of a dataset.

    The following metrics are supported: Silhouette, Calinski-Harabasz, Davies Bouldin, Elbow and connectivity.

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of cluster to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)
    :param display: if True, a plot of the scores will be displayed
    :param col_wrap: the number of columns/plots on each row of the plot showing the results of all scores (used if the
        results are displayed)
    :param score_metrics: a list containing the methods to use to determine the number of clusters.
        By default: Silhouette, Calinski-Harabasz, Davies Bouldin, Elbow and connectivity are used
    :param kwargs: additional parameters that could be sent to the seaborn catplot function to customize the plot

    :return: a dataframe containing all scores for all the numbers of clusters tested (for the different methods)
    """
    if score_metrics is None:
        score_metrics = ['silhouette_score',
                         'calinski_harabasz_score',
                         'davies_bouldin_score',
                         'connectivity_score',
                         'elbow_score']
    if not isinstance(score_metrics, list):
        score_metrics = [score_metrics]
    if param is None:
        param = {}

    metrics_details = {
        'silhouette_score': compute_silhouette_score,
        'calinski_harabasz_score': compute_calinski_harabasz_score,
        'davies_bouldin_score': compute_davies_bouldin_score,
        'connectivity_score': compute_connectivity_score,
        'elbow_score': compute_elbow_score
    }
    scores = pd.DataFrame({"number_clusters_tested": range(2, max_number_clusters + 1)})
    for metric in score_metrics:
        score = metrics_details[metric](data=data,
                                        max_number_clusters=max_number_clusters,
                                        algorithm=algorithm,
                                        param=param)
        scores = scores.merge(pd.DataFrame(score), on="number_clusters_tested")

    if display:
        plot_find_optimal_number_of_clusters(scores, col_wrap, **kwargs)

    return scores


def plot_find_optimal_number_of_clusters(scores, col_wrap=3, **kwargs):
    """
    Print the results of the "find_optimal_number_of_clusters" function

    :param scores: a dictionary containing the results of all methods for all the number of clusters tested
    :param col_wrap: the number of columns/plot on each row of the plot showing the results of all scores
    :param kwargs: additional parameters that could be sent to the seaborn catplot function to customize the plot
    """
    long_scores = pd.melt(scores,
                          id_vars='number_clusters_tested',
                          value_vars=[col for col in scores.columns if "_score" in col],
                          value_name='score',
                          var_name='method')
    with sns.plotting_context(font_scale=1.5):
        g = sns.catplot(data=long_scores, y="score", x="number_clusters_tested", col="method", kind="bar",
                        col_wrap=col_wrap, sharex=False, sharey=False, color=sns.color_palette("Paired")[1], **kwargs)
        g.fig.subplots_adjust(top=0.85)
        g.set_titles("{col_name}", size=15)
        g.fig.suptitle('Silhouette (highest), Calinski-Harabasz (highest), Davies Bouldin (lowest), '
                       '\nConnectivity (lowest) and Elbow (the point of inflection on the curve)', size=17)
        plt.tight_layout()
        plt.show()


def compute_elbow_score(data, max_number_clusters=9, algorithm=KMeans, param=None):
    """
    Determine the ideal number of clusters using the Elbow score

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of clusters to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)

    :return: a dictionary containing all scores for all the numbers of clusters tested
    """
    if param is None:
        param = {}

    return __apply_one_score(data=data, max_number_clusters=max_number_clusters, algorithm=algorithm, param=param,
                             metric_name='elbow_score')


def compute_silhouette_score(data, max_number_clusters=9, algorithm=KMeans, param=None):
    """
    Determine the ideal number of clusters using the Silhouette score

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of clusters to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)

    :return: a dictionary containing all scores for all the numbers of clusters tested
    """
    if param is None:
        param = {}

    return __apply_one_score(data=data, max_number_clusters=max_number_clusters, algorithm=algorithm, param=param,
                             metric_to_run=metrics.silhouette_score, metric_name='silhouette_score')


def compute_calinski_harabasz_score(data, max_number_clusters=9, algorithm=KMeans, param=None):
    """
    Determine the ideal number of clusters using the Calinski-Harabasz score

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of clusters to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)

    :return: a dictionary containing all scores for all the numbers of clusters tested
    """
    if param is None:
        param = {}

    return __apply_one_score(data=data, max_number_clusters=max_number_clusters, algorithm=algorithm, param=param,
                             metric_to_run=metrics.calinski_harabasz_score, metric_name='calinski_harabasz_score')


def compute_davies_bouldin_score(data, max_number_clusters=9, algorithm=KMeans, param=None):
    """
    Determine the ideal number of clusters using the Davies Bouldin score

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of clusters to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)

    :return: a dictionary containing all scores for all the numbers of clusters tested
    """
    if param is None:
        param = {}

    return __apply_one_score(data=data, max_number_clusters=max_number_clusters, algorithm=algorithm, param=param,
                             metric_to_run=metrics.davies_bouldin_score, metric_name='davies_bouldin_score')


def compute_connectivity_score(data, max_number_clusters=9, algorithm=KMeans, param=None):
    """
    Determine the ideal number of clusters using the connectivity score

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param max_number_clusters: the maximal number of clusters to be tested
    :param algorithm: the algorithm to be used; it must be a scikit-learn clustering function (see test function)
    :param param: the parameters of the algorithm (as a dictionary)

    :return: a dictionary containing all scores for all the numbers of clusters tested
    """
    if param is None:
        param = {}

    return __apply_one_score(data=data, max_number_clusters=max_number_clusters, algorithm=algorithm, param=param,
                             metric_to_run=_calculate_connectivity_score, metric_name='connectivity_score')


def _calculate_connectivity_score(data, labels, neighbourhood=10, metric='euclidean'):
    """
    Compute the connectivity score

    Connectivity captures the degree to which data points are connected
    within a cluster by keeping track of whether neighboring points are put
    into the same cluster. For each point, it checks all the neighboring points
    and adds a penalty if that point is not in the same cluster, i.e. it
    indicates the degree of connectedness of the clusters.

    source: https://cran.microsoft.com/web/packages/clValid/vignettes/clValid.pdf

    :param data: the dataset (i.e. a vector containing all the features needed for the clustering)
    :param labels: array containing for each row, the assigned classification label
    :param neighbourhood: the amount of neighbours to use in this method
    :param metric: the metric to use when calculating distance between neighboring points
        in a feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or a
        metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS

    :return: the connectivity score
    """
    # Make sure that the index starts at 0 (e.g. in case of sampling)
    data = pd.DataFrame(data)
    data = data.reset_index(drop=True)

    # Calculate distance matrix
    distance_matrix = pairwise_distances(data, metric=metric)
    score = 0
    for index, observation in data.iterrows():
        sorted_indexes = np.argsort(distance_matrix[index])
        current_label = labels[index]
        for neighbour in range(1, neighbourhood+1):
            neighbour_label = labels[sorted_indexes[neighbour]]
            if neighbour_label != current_label:
                score += 1 / neighbour

    return score
