# ©, 2019, Sirris
# owner: HCAB

"""Functions for outlier detection"""

import numpy as np
from statsmodels.stats.stattools import medcouple


def boxplot_outliers(x,
                     alpha=3,
                     max_sample=-1,
                     correct_skewness=False,
                     mc=10000):
    """
    Label outliers based on the boxplot definition: median +- alpha * IQR.

    IQR is the difference between the 75th and 25th percentiles. The outlier
    definition for skewed data is taken from
    M. Hubert and S. Van der Veeken 2007, Outlier detection for skewed data.

    :param x : list or array of values
    :param alpha : threshold for lower and upper boundary definition
    :param max_sample : int. Maximum sample size allowed. If dataset is larger
    than max_sample, take a random sample. Default : -1 (use all)
    :param correct_skewness : bool. Whether to correct skewness in the data.
    Default: False
    :param mc : int. Calculate medcouple over a random sample this size.
    Default: 10000

    Returns
    -------
    list of booleans indicating whether value is outlier or not

    """
    if (max_sample != -1) & (len(x) > max_sample):
        x = np.random.choice(x, max_sample)

    # get percentiles and IQR
    x_percentile = np.nanpercentile(x, [25, 75])
    iqd = x_percentile[1] - x_percentile[0]

    if correct_skewness:
        mc = medcouple(np.random.choice(x, mc))
        if mc > 0:
            mc_correct = (-4 * mc, 3 * mc)
        else:
            mc_correct = (-3 * mc, 4 * mc)
    else:
        mc_correct = (0, 0)

    # define outlier bounds
    lower_bound = x_percentile[0] - alpha * np.exp(mc_correct[0]) * iqd
    upper_bound = x_percentile[1] + alpha * np.exp(mc_correct[1]) * iqd

    outlier_labels = [(x0 < lower_bound) | (x0 > upper_bound) for x0 in x]
    return outlier_labels, (lower_bound, upper_bound)
