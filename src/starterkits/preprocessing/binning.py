# ©, 2019, Sirris
# owner: HCAB

"""Binning functions"""

import pandas as pd
import numpy as np


def bin_variable(x, bins=20, bins_dict=None, ret_bins=False):
    """
    Bin pandas series into equally spaced n_bins

    :param x : pandas series.
    :param bins : int. Number of bins to segment x into. It can also be a list
    or an array of bin intervals. Default : 20
    :param bins_dict : dict. Dictionary with bin values and labels. Default :
    None.
    :param ret_bins : bool. Return bin values and labels? Default : False.

    Returns
    -------
    new binned pandas series
    bins: dict. Bin values and labels

    """
    if bins_dict is None:
        if (isinstance(bins, list)) | (isinstance(bins, np.ndarray)):
            x_bin = bins
        else:
            # define bins
            x_bin = np.linspace(x.min(), x.max(), bins)

    else:
        x_bin = bins_dict['values']

    # bin series
    x_cut = pd.cut(x, x_bin, include_lowest=True)
    x_cut = x_cut.apply(lambda xx: xx.mid).astype(np.float64)

    if ret_bins:
        return x_cut, {'values': x_bin}
    else:
        return x_cut


def bin2d(df, col1, col2, nbins):
    """
    Bin two variables of one dataframe in 2D space.

    The bins, bin IDs and bin counts are added to the input dataframe

    :param df: dataframe.
    :param col1: str. First variable
    :param col2: str. Second variable
    :param nbins : int. Number of bins to segment the two variables into. Can
    also be a tuple with the number of bins for each variable or a tuple with
    the bin intervals for each variable.

    Returns
    -------
    df: input dataframe with bin labels, IDs and counts added to it
    h2: tuple. Output of np.histogram2d (with matrix flipped)
    df_bins : dataframe. Dataframe with all bins and associated counts (even
    those with 0 count)
    bin_specs : dictionary. Dictionary with bin labels, IDs and intervals per
    variable
    bin_cols : dictionary. Dictionary with 2 element list with the name of
    the bin labels and bin ID columns for each variable

    """
    # bin the dataset
    h2 = np.histogram2d(df[col1], df[col2], bins=nbins)

    # flip the matrix
    h2 = (h2[0].T, h2[1], h2[2])

    # make meshgrids with bin labels (average between successive bin limits)
    bin_labels = [(b[1:] + b[: -1]) / 2 for b in h2[1:]]
    bin_mesh = np.meshgrid(bin_labels[0], bin_labels[1])

    # start dataframe with count for each bin
    df_bins = pd.DataFrame({'count': np.ravel(h2[0])})

    bin_cols = {}
    bin_specs = {}
    # loop through each variable and process bins
    for k, v in enumerate([col1, col2]):
        # define bin columns
        bin_cols[v] = ['bin_%s' % v, 'bin_%s_id' % v]

        # add bin label and ID to df_bins
        df_bins[bin_cols[v][0]] = np.ravel(bin_mesh[k])
        df_bins[bin_cols[v][1]] = df_bins.groupby(bin_cols[v][0]).ngroup()

        # bin values in main dataset
        df[bin_cols[v][0]] = pd.cut(
            df[v], bins=h2[k + 1], labels=bin_labels[k], include_lowest=True)
        df[bin_cols[v][0]] = df[bin_cols[v][0]].astype(float)

        # collect bin specs
        bin_specs0 = df_bins[bin_cols[v]].\
            drop_duplicates().\
            sort_values(bin_cols[v][1])
        bin_specs[v] = {}
        bin_specs[v][bin_cols[v][0]] = bin_specs0[bin_cols[v][0]].values
        bin_specs[v][bin_cols[v][1]] = bin_specs0[bin_cols[v][1]].values
        bin_specs[v]['bin_%s_intervals' % v] = h2[k + 1]

    # add counts and IDs for each bin in main dataset
    df = df.drop(['count', bin_cols[col1][1], bin_cols[col2][1]],
                 axis=1,
                 errors='ignore').\
        merge(df_bins, on=[bin_cols[col1][0], bin_cols[col2][0]], how='left')

    return df, h2, df_bins, bin_specs, bin_cols
