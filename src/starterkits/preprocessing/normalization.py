# ©, 2019, Sirris
# owner: HCAB

"""Tools for normalization."""

import types

import numpy as np
import pandas as pd


def normalize(df,
              cols_to_normalize=None,
              group_col=None,
              method="z_score",
              in_self=False,
              prefix='n_'):
    """
    Normalize a dataset, optionally group-by-group, using the specified

    normalization method.

    :param df: the Pandas DataFrame containing the dataset to normalize
    :param cols_to_normalize: a list referencing the columns of the DataFrame to
    normalize; non-numeric columns will be dropped; if None (default), normalize
    all columns
    :param group_col: a string with the name of the column to group-by
    :param method: the normalization method to use; the following values are
    supported: 'minmax', 'z_score' (default), and 'max'; you can also pass a
    function
    :param in_self: a Boolean indicating whether to add the normalized columns
    to the input DataFrame (default value: False)
    :param prefix: str. Prefix to append to column names. Default: 'n_'

    :returns: the normalized DataFrame
    """
    if isinstance(method, (types.LambdaType, types.FunctionType)):
        method_func = method
    else:
        try:
            method_func = {
                "minmax": lambda x: minmax(x),
                "z_score": lambda x: z_score(x),
                "max": lambda x: x / x.max()
            }[method]
        except KeyError:
            raise ValueError(
                "Unsupported method " + str(method) +
                "; must be one of 'minmax', 'z_score', 'max', or a function")

    if cols_to_normalize is None:
        # drop non-numeric columns
        cols_to_normalize = list(df.select_dtypes(['number']).columns)
    elif isinstance(cols_to_normalize, str):
        cols_to_normalize = [cols_to_normalize]
    elif type(cols_to_normalize) != list:
        raise ValueError("cols_to_normalize must be a list")

    if group_col is not None:
        d_out = (df.groupby(group_col)[cols_to_normalize]
                 .transform(method_func))
    else:
        d_out = df[cols_to_normalize].transform(method_func)

    if in_self:
        d_out.columns = [prefix + c for c in d_out.columns]
        df = pd.concat([df, d_out], axis=1)

        return df
    else:
        return d_out


def z_score(x):
    """
    Apply z-score normalization to array.

    :param x: a NumPy array

    :returns: array with the normalized values
    """
    return (x - x.mean()) / x.std()


def minmax(x):
    """
    Apply minmax normalization to array.

    :param x: a NumPy array

    :returns: array with the normalized values
    """
    return (x - x.min()) / (x.max() - x.min())


def replace_nans(x, aggfun=np.nanmean, min_nonnan=None):
    """
    Replace missing values (NaNs) with column average.

    :param x: a 2D array with assets as rows and timestamp as columns; can also
    be a Pandas DataFrame, in which case it will first be converted to an array
    and re-converted back to a DataFrame in the end
    :param aggfun: a function to use in column aggregation (default: np.nanmean)
    :param min_nonnan: an int indicating the minimum number of non-null values
    per column for replacement to be made (default: None)

    :returns: an array (or DataFrame) with NaNs replaced by column average
    """
    converted_to_df = False
    x_cols = None
    x_idx = None

    if isinstance(x, pd.DataFrame):
        converted_to_df = True
        x_cols = x.columns
        x_idx = x.index
        x = x.values

    col_agg = aggfun(x, axis=0)
    id_nan = np.where(np.isnan(x))

    if min_nonnan is not None:
        ct_nan = (~np.isnan(x)).sum(axis=0)
        col_agg[ct_nan < min_nonnan] = np.nan
    x[id_nan] = col_agg[id_nan[1]]

    if converted_to_df:
        x = pd.DataFrame(x, columns=x_cols)
        x.set_index(x_idx, inplace=True)

    return x
