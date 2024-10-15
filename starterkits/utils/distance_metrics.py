# ©, 2022, Sirris
# owner: FFNG
""" Distance metrics used for anomaly detection. """
import numpy as np
from sklearn.metrics import pairwise
from scipy.spatial.distance import mahalanobis


def frobenius_norm(a, b):
    """ Frobenius norm of the differences. """
    return np.linalg.norm(a.sort_index() - b.sort_index(), ord='fro')


def frobenius_norm_v2(a, b):
    """ Power of the Frobenius norm of the differences. """
    return np.linalg.norm(a.sort_index() - b.sort_index(), ord='fro')**2


def frobenius_norm_v3(a, b):
    """ Square root of the Frobenius norm of the differences. """
    return np.sqrt(np.linalg.norm(a.sort_index() - b.sort_index(), ord='fro'))


def matrix_distance_(dist_function, A, B):
    """ Given two matrices A and B, transform them to vectors and calculate their pairwise distance. """
    return dist_function(
        A.to_numpy().flatten().reshape(1, -1),
        B.to_numpy().flatten().reshape(1, -1)
    )[0][0]


def cosine_distance(A, B):
    """ Cosine distance of two matrices A and B. """
    return matrix_distance_(pairwise.cosine_distances, A, B)


def manhattan_distance(A, B):
    """ Manhattan distance of two matrices A and B. """
    return matrix_distance_(pairwise.manhattan_distances, A, B)


def mahalanobis_distance(A, B, SI):
    """ Mahalanobis distance of two matrices A and B given the inverse of the co-variance matrix (SI)."""
    # unstack matrices A and B
    a = A.to_numpy().flatten().reshape(1, -1)    # !!! 96 dimensions (each sensor)
    b = B if isinstance(a, np.ndarray) else B.to_numpy()
    # b = B#.flatten().reshape(1, -1)  # !!! 16 dimensions (only mean of sensor)
    dist_ = mahalanobis(a, b, SI)
    return dist_
