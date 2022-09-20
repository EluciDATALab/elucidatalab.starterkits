# ©, 2019, Sirris
# owner: HCAB

"""Collection of distance functions"""

import numpy as np
import re
# these following functions are read by function "distance"
from starterkits.statistical_tools.angular import get_angular_distance


def mixed_type_pairwise_euclidean(x, y, types, ang_boundary=1):
    """
    Calculate pairwise distance between two input arrays. Distance metric

    depends on data type.

    :param x : array. 2D array with shape RxC
    :param y : array. 2D array with shape RxC2
    :param types : list. List of data types
    :param ang_boundary: float. Boundary value to be passed to
    get_angular_distance

    :returns distance
    """
    dist = np.zeros((x.shape[1], y.shape[1], len(types)))
    for k, i in enumerate(types):
        x0, y0 = np.meshgrid(x[k, :], y[k, :])
        if re.match('num', i.lower()):
            dist[:, :, k] = np.abs(x0 - y0).T
        elif re.match('ang', i.lower()):
            dist[:, :, k] = \
                get_angular_distance(x0, y0, boundary=ang_boundary).T

    return dist.sum(axis=2)


def euclidean_coordinates(xy1, xy2):
    """
    Calculates all pair-wise distances between two sets of points

    :param xy1 : array. 2D coordinates array (points x X/Y coordinates)
    :param xy2 : array. 2D coordinates array (points x X/Y coordinates)

    :returns distance matrix

    """
    x1, x2 = np.meshgrid(xy1[:, 0], xy2[:, 0])
    y1, y2 = np.meshgrid(xy1[:, 1], xy2[:, 1])

    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    return dist


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth

    (specified in decimal degrees) in kilometers
    from https://stackoverflow.com/questions/4913349/haversine-formula-in-
    python-bearing-and-distance-between-two-gps-points

    :param lon1: array. 1D array of longitude values of first set of coordinates
    :param lat1: array. 1D array of latitude values of first set of coordinates
    :param lon2: array. 1D array of longitude values of second set of
    coordinates
    :param lat2: array. 1D array of latitude values of second set of coordinates

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r
