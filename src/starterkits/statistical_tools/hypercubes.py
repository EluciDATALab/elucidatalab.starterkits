# ©, 2019, Sirris
# owner: AMUR

"""
Collection of functions to perform hypercube operations

Existing functions:
- hypercube_analysis
- define_cubes_in_df
- define_binning
- get_nearest_cube
"""

from functools import reduce

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from starterkits.preprocessing.binning import bin_variable
from starterkits.preprocessing.outliers import boxplot_outliers


def hypercube_analysis(df, ref_vars, target,
                       cube_specs=None,
                       cube_ids=None,
                       bins=20,
                       lower_bound=None,
                       upper_bound=None):
    """
    Perform hypercube analysis by binning dataset into its reference variables

    and deriving standard statistics on each bin

    :param df: dataframe. Main dataset containing target and reference variables
    :param ref_vars: list. List of reference variable(s) to use.
    :param target: str. Name of column with target.
    :param cube_specs: dict. Dictionary with specs for binning. Dictionary
    should have one entry per reference
    variable with keys 'values' (bin intervals) and 'labels' (bin labels).
    Default : None.
    :param cube_ids: dataframe. Dataframe with one column with bin labels per
    reference variable and a column specifying the ID of that hypercube.
    :param  bins: int. Number of bins to cut reference variable in to. It can
    also be a list or array with bin intervals or a dictionary with the bins per
    reference variable. Default : 20.
    :param lower_bound: method. Determines the lower and upper bound of a cube.
    Default is None (will use default boxplot_outliers bounds)
    :param upper_bound: see lower_bound

    :returns: cubes: dataframe. Hypercubes statistical summary.
              cube_specs: dictionary with specs on cubes

    """
    # get binning
    if cube_specs is None:
        binning, cube_specs, _ = define_binning(df, ref_vars, bins)
    else:
        binning = [bin_variable(df[b], bins_dict=cube_specs[b])
                   for b in cube_specs.keys()]
        # bin_keys = list(cube_specs.keys())

    # apply binning to fleet summary frame
    if lower_bound is None:
        def lower_bound(x):
            return boxplot_outliers(x)[1][0]
    if upper_bound is None:
        def upper_bound(x):
            return boxplot_outliers(x)[1][1]
    # get stats per cube
    cubes = df.groupby(binning)[target]. \
        agg(['median', 'mean', 'sum', 'count', 'min', 'max',
             lower_bound, upper_bound]). \
        reset_index(). \
        sort_values('mean', ascending=False)

    # label cubes
    if cube_ids is not None:
        cubes = cubes.merge(cube_ids, on=list(ref_vars))
    else:
        cubes['cube_id'] = np.arange(len(cubes))

    return cubes, cube_specs


def extract_cubes(df,
                  ref_vars,
                  target_vars,
                  cube_specs,
                  lower_bnd,
                  upper_bnd,
                  cube_ids,
                  nbins):
    """
    Given a dataframe df, extract the hypercubes based on ref_var (e.g. wind

    speed) and target var (e.g. active power). This function is just a wrapper
    for hypercubes.hypercube_analysis where some input variables are already set.

    :param df: dataframe. Main dataset containing target and reference variables
    :param ref_vars: list. List of reference variable(s) to use.
    :param target_vars:  str. Name of column with target.
    :param cube_specs: dict. Dictionary with specs for binning
    :param lower_bnd: number. lower bound of the percentile
    :param upper_bnd: number. upper bound of the percentile
    :param cube_ids: dataframe. Dataframe with one column with bin labels per
    reference variable and a column specifying the ID of that hypercube.
    :param nbins:  int. Number of bins to cut reference variable in to.
    It can also be a list or array with bin intervals or a dictionary with the
    bins per reference variable.

    :return: cubes : dataframe. Hypercubes statistical summary
    :return: cube_ids: dataframe. Dataframe with one column with bin labels per
    :return: cube_specs : dictionary with specs on cubes
    :return: cubes_reduced : dataframe. Same as cubes but reports only columns
    that refer to ref_var, cube_id, target median

    """
    def lower_bound(x):
        return np.nanpercentile(x, lower_bnd)

    def upper_bound(x):
        return np.nanpercentile(x, upper_bnd)

    cubes_reduced = []
    cubes = []

    for t in target_vars:
        cubes, cube_specs = hypercube_analysis(df, ref_vars, t, cube_specs=cube_specs, cube_ids=cube_ids, bins=nbins,
                                               lower_bound=lower_bound, upper_bound=upper_bound)
        if cube_ids is None:
            cube_ids = cubes[np.append(ref_vars, 'cube_id')].drop_duplicates()

        cubes_reduced.append(
            cubes.rename(columns={'median': 'cube_%s' % t})[np.append(np.append(ref_vars, 'cube_id'), 'cube_%s' % t)])

    cubes_reduced = reduce(lambda x, y: x.merge(y, on=list(np.append(ref_vars, 'cube_id'))), cubes_reduced)

    return cubes_reduced, cube_specs, cube_ids, cubes


def define_binning(df, ref_vars, bins):
    """
    Apply binning across a set of variables

    :param df : dataframe. Main dataset containing reference variables
    :param ref_vars: list. List of reference variable(s) to use.
    :param bins : int. Number of bins to cut reference variable in to.
    It can also be a list or array with bin intervals or a dictionary with the bins per
    reference variable. Default : 20.

    Returns
    -------
    binning : list. list with objects to be passed to pd.DataFrame.groupby. One
    object per reference variable
    cube_specs : dictionary with specs on cubes

    """
    if type(bins) != dict:
        bins = {f: bins for f in ref_vars}
    bins = {f: bin_variable(df[f], bins=bins[f], ret_bins=True)
            for f in ref_vars}
    binning = [bins[b][0] for b in bins.keys()]
    # bin_keys = list(bins.keys())
    cube_specs = {b: bins[b][1] for b in bins.keys()}

    return binning, cube_specs, bins


def define_cubes_in_df(df, ref_vars, target_var, bins=20,
                       cube_specs=None, cube_ids=None,
                       order_cubes=True, drop_stats=False,
                       col_index=None, assign_nearest_cube=False):
    """
    Add cube_id, to dataset and return as well median target var and count of cube occurrences.

    Binning specs used in cube definition are also returned

    :param df : dataframe. Main dataset containing reference variables
    :param ref_vars: list. List of reference variable(s) to use.
    :param target_var : str. Name of column with target.
    :param bins : int. Number of bins to cut reference variable in to. It can also
    be a list or array with bin intervals or a dictionary with the bins per
    reference variable. Default : 20.
    :param cube_specs : dict. Dictionary with specs for binning. Dictionary
    should have one entry per reference variable with keys 'values' (bin
    intervals) and 'labels' (bin labels). Default : None.
    :param cube_ids : dataframe. Dataframe with one column with bin labels per
    reference variable and a column specifying the ID of that hypercube.
    :param order_cubes : bool. Whether to sort cubes by median target_var. Default : True
    :param drop_stats : bool. Whether to drop cube stats. Default : False.
    :param col_index : str. Name of index. If none, ignore index. Default: None
    :param assign_nearest_cube : bool. Whether to find nearest cube for entries in the
    dataset with no cube assigned. Needs cube_ids input. Default : False

    :returns dataframe : input dataset with extra column for cube_id
    :return cube_specs : dictionary with specs on cubes

    """
    if cube_specs is None:
        # define hypercube binning
        _, cube_specs, bins = define_binning(df, ref_vars, bins)
    else:
        bins = None

    # loop through reference variables and apply binning
    cube_vars = []
    cube_specs = cube_specs or {}
    for f in ref_vars:
        binning = bin_variable(
            df[f],
            bins=bins[f] if bins is not None else None,
            bins_dict=cube_specs[f] if cube_specs is not None else None,
            ret_bins=True)
        cube_specs[f] = binning[1]
        df['cube_%s' % f] = binning[0].values
        cube_vars.append('cube_%s' % f)

    # get bin stats
    cubes = df.groupby(cube_vars)[target_var]. \
        agg(['median', 'count']). \
        reset_index()
    if cube_ids is None:
        # order cubes and label
        if order_cubes:
            cubes.sort_values('median', ascending=False, inplace=True)
        cubes['cube_id'] = np.arange(len(cubes))
    else:
        renamer = {v: 'cube_%s' % v for v in cube_ids.columns
                   if v != 'cube_id'}
        cubes = cubes.merge(cube_ids.rename(columns=renamer), on=cube_vars)

    # add to main dataset
    if col_index is not None:
        df.reset_index(inplace=True)
    df = df.drop(['median', 'count', 'cube_id'], axis=1, errors='ignore'). \
        merge(cubes, on=cube_vars, how='left'). \
        drop(cube_vars, axis=1)

    if (cube_ids is not None) & (assign_nearest_cube is True):
        if df.cube_id.isnull().sum() > 0:
            df = get_nearest_cube(cube_ids, df, ref_vars)
            df = df.drop(['median', 'count'], axis=1). \
                merge(cubes, on='cube_id', how='left')
    if drop_stats:
        df.drop(['median', 'count'], axis=1, inplace=True)

    if col_index:
        df.set_index(col_index, inplace=True)

    return df, cube_specs


def get_nearest_cube(cubes, ds, ref_vars):
    """
    For entries that could not be assigned to any of the input cubes:

    identify the nearest (euclidean distance) cube and assign tha entry to it.
    Variables will be normalized by the maximum variable value before distance
    computation

    :param cubes : dataframe. Dataframe containing cube_ids and reference variables
    :param ds : dataframe. Main dataset containing reference variables
    :param ref_vars: list. List of reference variable(s) to use.

    :returns input dataset with cube_ids assigned

    """
    # calculate euclidean distances
    cube_dims = cubes[ref_vars].values.T
    missing_cubes = ds.loc[ds.cube_id.isnull(), ref_vars]
    missing_cubes_idx = missing_cubes.index
    missing_cubes = missing_cubes.values.T
    # normalize
    both = np.concatenate([cube_dims, missing_cubes], axis=1)
    cube_dims = cube_dims / both.max(axis=1)[:, None]
    missing_cubes = missing_cubes / both.max(axis=1)[:, None]
    dist = euclidean_distances(missing_cubes.T, cube_dims.T)

    # define dataframe with nearest cube and distance
    nearest = pd.DataFrame({'cube_id': cubes.iloc[np.argmin(dist, axis=1)].
                           cube_id,
                            'dist': np.min(dist, axis=1)})
    nearest.set_index(missing_cubes_idx, inplace=True)

    # add to dataset
    ds = ds.join(nearest.rename(columns={'cube_id': 'nearest_cube'}))
    ds['cube_id'] = ds.cube_id.combine_first(ds.nearest_cube)
    ds.drop(['nearest_cube', 'dist'], axis=1, inplace=True)

    return ds


def estimate_output(df, ref_vars, target, col_index=None, hypercube_kwargs=None, estimator='median'):
    """
    Estimate the value of a target variable using hypercubes on a set of reference variables.

    This estimates will
    be added to the input dataset, together with the lower and upper boundaries and the residual

    :param df: dataframe. input dataframe containing target and reference variables
    :param ref_vars: list. set of reference variables
    :param target: str. Name of target variable to compute hypercubes on
    :param hypercube_kwargs: dict. Further params to be passed to hypercube_analysis
    :param col_index : str. Name of index. If none, ignore index. Default: None
    :param estimator: str. Function to be used to estimate target value. Either 'median' (default) or 'mean'
    :return: input dataframe with estimated target variable, lower and upper bounds and the residuals
    """
    if hypercube_kwargs is None:
        hypercube_kwargs = {}

    cubes, cube_specs = hypercube_analysis(df.copy(), ref_vars, target, **hypercube_kwargs)

    df, _ = define_cubes_in_df(df, ref_vars, target, cube_specs=cube_specs, col_index=col_index)

    if col_index is not None:
        df.reset_index(inplace=True)
    df = df.drop(columns=['median', 'count']).merge(cubes[['cube_id', estimator, 'lower_bound', 'upper_bound']],
                                                    on='cube_id',
                                                    how='left')
    if col_index is not None:
        df.set_index(col_index, inplace=True)
    df.rename(columns={estimator: '%s_estimated' % target}, inplace=True)
    df.drop(columns='cube_id', inplace=True)

    df['%s_residual' % target] = df[target] - df['%s_estimated' % target]

    return df


def extract_residuals(df, cubes, cube_specs, cube_ids, ref_vars, target_vars, col_index, res_prefix='res',
                      assign_nearest_cube=True, drop_cube_target_vars=True):
    """
    Given the df and the hypercubes, it computes for each row of the df the residual

    (namely the distance from the expected output) This residuals are provided as new dataframe (df_res)
    This function is just a wrapper for define_cubes_in_df where some input variables are already set

    :param df: dataframe. Main dataset containing reference variables
    :param cubes: dataframe. Hypercubes statistical summary.
    :param cube_specs: dictionary with specs on cubes
    :param cube_ids: dataframe. Dataframe with one column with bin labels per
    :param ref_vars: list. List of reference variable(s) to use.
    :param target_vars: list. List of target variable(s) to use.
    :param col_index: str. Name of the column that we want to promote as index
    :param res_prefix: str. Prefix to be used to indicate the column with residuals
    :param assign_nearest_cube: bool. Whether to find nearest cube for entries in the
    dataset with no cube assigned. Needs cube_ids input. Default : True
    :param drop_cube_target_vars: bool. If negative the dataframe contains also the expected value of the target_vars
    :return: dataframe. Main dataset + residual

    """
    df_res, _ = define_cubes_in_df(df.copy(), ref_vars, target_vars[0], drop_stats=True, col_index=col_index,
                                   cube_specs=cube_specs, cube_ids=cube_ids, assign_nearest_cube=assign_nearest_cube)

    df_res = df_res.reset_index().merge(cubes.drop(ref_vars, axis=1), on='cube_id', how='left').set_index(col_index)

    for t in target_vars:
        cube_target_var = 'cube_%s' % t
        df_res[res_prefix + '_%s' % t] = df_res[cube_target_var] - df_res[t]

        if drop_cube_target_vars:
            df_res = df_res.drop(columns=cube_target_var)

    return df_res


def build_hypercube_model(df, ref_vars, target_vars, minimum_points, nbins, col_index):
    """
    This function is just a wrapper of the functions (1) hypercubes.extract_cubes and (2) hypercubes.extract_residuals.

    Given a dataframe, the function:
    - extracts the hypercubes based on ref_var (e.g. wind speed, wind direction) and target var (e.g. active power)
    - once the hypercube are extracted, it computes for each hypercube the residuals
    (namely the difference between expected and real value of target var)

    Args:
    :param df: dataframe. Main dataset containing target and reference variables
    :param ref_vars: list. List of reference variable(s) to use.
    :param target_vars:  str. Name of column with target.
    :param minimum_points: minimum number of point per cube to retain the hypercube.
        Cubes with less points are not used.
    :param nbins: int. Number of bins to cut reference variable in to.
            It can also be a list or array with bin intervals or a dictionary with the bins per
    reference variable.
    :param col_index: str. Name of the column that we want to promote as index


    Returns
    -------
    df_res : input dataset with extra column containing the expected value of the target_var and the residuals
    (difference between expected and real value)
    cubes_reduced, cube_specs, cube_ids, cubes0 : all objects provided by the method extract_cubes
    (provided that the cubes satisfy the condition minimum_points)

    """
    df_copy = df.copy(deep=True)
    cubes_reduced, cube_specs, cube_ids, cubes0 = extract_cubes(df_copy, ref_vars, target_vars,
                                                                None, 40, 60, None, nbins)

    # Filter cubes with a number of points that is below minPoint
    few_points_cubes = cubes0[cubes0['count'] <= minimum_points].cube_id.values
    cubes_reduced = cubes_reduced[~cubes_reduced.cube_id.isin(few_points_cubes)]
    cube_ids = cube_ids[~cube_ids.cube_id.isin(few_points_cubes)]

    res_prefix = 'res'
    df_res = extract_residuals(df_copy,
                               cubes_reduced, cube_specs, cube_ids, ref_vars,
                               target_vars, col_index, res_prefix)

    column_residual = res_prefix + '_' + target_vars[0]
    column_expected_target_var = target_vars[0] + '_expected'
    df_res[column_expected_target_var] = df_res[target_vars[0]] + df_res[column_residual]

    return df_res, cubes_reduced, cube_specs, cube_ids, cubes0
