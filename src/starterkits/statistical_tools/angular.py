# ©, 2019, Sirris
# owner: HCAB

"""
A collection of functions to easily operate on angular data. Most of them are

simple wrappers around regular modules.

Existing functions:
- get_angular_distance
- circular_mean
- circular_std
- rotate_points
"""

import math

import numpy as np
from scipy import stats
from geographiclib.geodesic import Geodesic


def get_angular_distance(x, y, boundary=360):
    """
    Get the smallest difference between two angles

    :param x : list, array or float. First angle(s)
    :param y : list, array or float. Second angle(s). Must have the same
    dimensions as x
    :param boundary : float. Angular data period value

    :returns: list
    """
    # convert to numpy arrays, if input is list
    x = np.array(x) if isinstance(x, list) else x
    y = np.array(y) if isinstance(y, list) else y

    return ((x - y) + boundary / 2) % boundary - boundary / 2


def circular_mean(x, unit='degree', **kwargs):
    """
    Get the circular mean of a set of angles.

    Wrapper around scipy.strats.circmean

    :param x : array like. Set of angles to calculate mean from.
    :param unit : str. Unit angles are in. Can be ['degrees','radians'].
    Default: 'degree'
    :param kwargs : further parameters to be passed to scipy.stats.circmean

    :returns: float. Mean of set of angles
    """
    if unit == 'degree':
        unit_high = 360
    elif unit == 'radians':
        unit_high = (2 * np.pi)
    else:
        raise Exception("Error, unit must be one of ['degree', 'radians']")

    x = [x0 for x0 in x if ~np.isnan(x0)]
    return stats.circmean(x, high=unit_high, **kwargs)


def circular_std(x, unit='degree', **kwargs):
    """
    Get the circular standard deviation of a set of angles.

    Wrapper around scipy.strats.circmean

    :param x : array like. Set of angles to calculate std from.
    :param unit : str. Unit angles are in. Can be ['degrees','radians'].
    Default: 'degree'
    :param kwargs : further parameters to be passed to scipy.stats.circstd

    :returns: float. Mean of set of angles
    """
    if unit == 'degree':
        unit_high = 360
    elif unit == 'radians':
        unit_high = (2 * np.pi)
    else:
        raise Exception("Error, unit must be one of ['degree', 'radians']")

    return stats.circstd(x, high=unit_high, **kwargs)


def rotate_points(origin, point, angle):
    """
    Rotate a set of points counterclockwise by a given angle around an origin.

    Look at [https://stackoverflow.com/questions/34372480/rotate-point-about-
    another-point-in-degrees-python]

    :param origin: tuple. xy coordinates of origin
    :param point: tuple. xy coordinates of points to be rotated
    :param angle: float. Rotation in degrees

    :returns: tuple with rotated points
    """
    ox, oy = origin
    px, py = point

    angle = np.deg2rad(angle)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def angle_between_points(p1, p2):
    """
    Get slope and angle between two points

    :param p1: tuple. xy coordinates of first point
    :param p2: tuple. xy coordinates of second point

    :returns: tuple slope and angle (degrees)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    if len(p1.shape) == 1:
        p1 = p1[None, :]
        p2 = p2[None, :]

    slope = (p2[:, 1] - p1[:, 1]) / (p2[:, 0] - p1[:, 0])

    angle = np.rad2deg(np.arctan(slope))

    return slope[0], angle[0]


def angle_between_coordinates(lat1, lon1, lat2, lon2):
    """
    Calculate angle between two points in respect to the North

    :param lat1: float. Latitude coordinates of first point
    :param lon1: float. Longitude coordinates of first point
    :param lat2: float. Latitude coordinates of second point
    :param lon2: float. Longitude coordinates of second point
    :return:
    """
    geod = Geodesic.WGS84

    azi = geod.Inverse(lat1, lon1, lat2, lon2)['azi1']
    return 360 - np.mod(360 - azi, 360)
