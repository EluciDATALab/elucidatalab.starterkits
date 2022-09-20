# ©, 2019, Sirris
# owner: HCAB

"""Tools for visualizing clusters/helping finding the number of clusters"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from starterkits.visualization import vis_tools


def plot_clustermap_precomputed_distance(distance_matrix,
                                         axes=None,
                                         draw_y_ticks=False,
                                         labels=None,
                                         n_labels=-1,
                                         dendrogram_cut_threshold=None,
                                         linkage_kwargs=None):
    """
    Plot a heatmap of the input distance matrix and a dendrogram calculated from it.

    Clustermap from seaborn provides a similar feature, however its utilization
    is not very straightforward (it includes the calculation of the linkage
    matrix and passing it as input). Furthermore, it is not possible to provide
    a threshold to cut the dendrogram at a given level and color all leaves
    below that threshold with a different color

    :param distance_matrix: array. Square matrix
    :param axes: axes object. Axes to make the plots in. Default: None (will be
     created)
    :param draw_y_ticks: bool. Whether to draw the y-ticks on the dendrogram
     (might conflict with the plot). Default: False
    :param labels: array/list. Labels for the x-axis ticks. Default: None
    :param n_labels: int. Which ticks to plot. Default: -1 (use all)
    :param dendrogram_cut_threshold: float. Value to prune the dendrogram tree.
     Branches underneath this threshold will have a unique color. Default: None
     (use dendrogram function default)
    :param linkage_kwargs: dict. Further arguments to pass to
     scipy.cluster.hierarchy.linkage
    """
    if linkage_kwargs is None:
        linkage_kwargs = {'metric': 'euclidean', 'method': 'complete'}
    if axes is None:
        fig, axes = plt.subplots(2, 1,
                                 figsize=(15, 20),
                                 gridspec_kw={'height_ratios': [1, 3]})

    # convert input data to square matrix
    dists = squareform(distance_matrix)

    # calculate linkage matrix
    linkage_matrix = linkage(dists, **linkage_kwargs)

    # plot dendrogram
    dendrogram(linkage_matrix,
               ax=axes[0],
               color_threshold=dendrogram_cut_threshold)

    xticks = [int(x.get_text()) for x in axes[0].get_xticklabels()]
    axes[0].set_xticks([])
    if draw_y_ticks:
        ymax = axes[0].get_ylim()[1]
        yticks = np.linspace(ymax/2, ymax, 5)
        axes[0].set_yticks(yticks)
        axes[0].set_yticklabels(['%0.1f' % y for y in yticks])
        axes[0].tick_params(axis="y", direction="in", pad=-30)
    else:
        axes[0].set_yticks([])

    # plot distance matrix as heatmap
    axes[1].imshow(distance_matrix[np.ix_(xticks, xticks)], aspect='auto')
    if labels is None:
        axes[1].set_xticks([])
    else:
        labels = labels[np.ix_(xticks)]
        id_labels = np.arange(len(labels)) if n_labels == -1 \
            else np.int64(np.linspace(0, len(labels) - 1, n_labels))
        axes[1].set_xticks(id_labels)
        axes[1].set_xticklabels([labels[lab] for lab in id_labels])
        vis_tools.format_axes(axes[1], rotate=90, is_date=False)
    axes[1].set_yticks([])

    vis_tools.axes_font_size('', '', 'Distance matrix', axes=axes[0])
    vis_tools.axes_font_size('', '', 'Dendrogram', axes=axes[1])

    plt.show()

    return axes
