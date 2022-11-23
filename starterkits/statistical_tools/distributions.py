# ©, 2019, Sirris
# owner: HCAB

"""
Collection of functions for density estimation

Existing functions:
- Kernel Density Estimation (KDE)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def kde_density_1d(x, x_range=None, npoints=1000, x_vec=None, kde_kwargs=None):
    """
    Calculate density curve for 1D data using sklearn KernelDensity method

    :param x: array. data to apply density calculation on
    :param x_range: tuple, array, list. Min and max values for kde estimation.
    Default: None (uses the min and max from x)
    :param npoints: int. Number of points for kde estimation. Default: 1000
    :param x_vec: array, list. Values at which to make kde estimation.
    Default: None (will be created from x_range and npoints)
    :param kde_kwargs: dict. Arguments to be passed KernelDensity estimation
    :returns: density: array. kde estimates
    :returns: x_vec_fit: array. Points at which kde was estimated
    """
    kde_kwargs = {} if kde_kwargs is None else kde_kwargs
    kde = KernelDensity(**kde_kwargs)

    x = x.values if isinstance(x, pd.Series) else x

    x_fit = x if x.ndim == 2 else x.reshape(-1, 1)
    kde.fit(x_fit)

    if x_vec is None:
        x_range = (np.min(x), np.max(x)) if x_range is None else x_range
        x_vec = np.linspace(*x_range, npoints)

    x_vec_fit = x_vec if x_vec.ndim == 2 else x_vec.reshape(-1, 1)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_vec_fit).squeeze()

    return np.exp(logprob), x_vec_fit.squeeze()
