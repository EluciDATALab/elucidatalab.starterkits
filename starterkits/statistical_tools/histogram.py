# ©, 2019, Sirris
# owner: AMUR

"""
A collection of functions to easily operate on histogram data.

Existing functions:
- intersection
"""

import numpy as np


# This code comes from stackoverflow (https://stackoverflow.com/a/52204682)
def histogram_intersection(hist_1, hist_2, bin_edges):
    """
    Compute the intersection between two histogram distributions.

    Although not required, it is convenient to pass normalized distributions
    when the two distributions are based on sample of different size.
    In case of normalized distributions the intersection is bounded between 1
    (perfect overlapping) and 0
    (no overlapping between the distributions)

    :param hist_1: array. First histogram (reports the (relative) frequency of
    samples in that bin)
    :param hist_2: array. Second histogram
    :param bin_edges:  array. Define the range of the bins.
    :return: intersection. float. The level of intersection between hist_1 and
    hist_2
    """
    bin_edges = np.diff(bin_edges)
    intersection = 0
    for i in range(len(bin_edges)):
        intersection += min(bin_edges[i] * hist_1[i], bin_edges[i] * hist_2[i])
    return intersection
