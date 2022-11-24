# ©, 2019, Sirris
# owner: HCAB

"""collection of functions to operate on dataframes"""


import pandas as pd
import numpy as np
from scipy.ndimage.measurements import label as label_connected
import itertools
from starterkits.utils.manager import progressbar


def label_connected_elements(x, target, time_col,
                             group=None,
                             label_col='labeled'):
    """
    Label consecutive events (possibly by group) using function label from

    scipy.ndimage.measurements

    :param x : dataframe. main dataset with events to label. The index of the
    dataframe should contain the timing variable
    :param target : str. Name of variable to use when labeling connected
    elements
    :param time_col : str. Sort dataset by this column
    :param group : str. Name of grouping variable in dataset. If none, apply on
    entire
    dataset. Default : None
    :param label_col : str. Name to use for the labeled column.
    Default: 'labeled'
    Returns
    ------
    array with labels of connected elements
    """
    if group is None:
        x['dummy'] = 1
        group = 'dummy'

    # sort by time
    x.sort_values(list(np.append(group, time_col)), inplace=True)
    x[label_col] = (x
                    .groupby(group)
                    [target]
                    .transform(lambda xx: label_connected(xx)[0]))

    if x[label_col].dtype == bool:
        x[label_col] = x[label_col].astype(int)

    x.drop(columns=['dummy'], inplace=True, errors='ignore')

    return x


def _expand_dataframe(x, r, group, time_col, target, t_int, check_max_rows):
    """
    Expand dataframe to have regular time intervals

    :param x : dataframe. main dataset with events to label. The index of the
    dataframe should contain the timing variable
    :param r: series. Series containing specification of a specific group
    :param target : str. Name of variable to use when labeling connected
    elements
    :param time_col : str. Sort dataset by this column
    :param t_int : timedelta. Median time difference between samples of the
    same group
    :param group : str. Name of grouping variable in dataset. If none, apply on
    entire dataset. Default : None
    :param check_max_rows: bool. Check number of rows after expansion and
    require user input if exceeded.
    :return dataframe with regularly spaced time intervals
    """
    global MAX_ROWS
    x0 = x.loc[tuple(r[group])]

    if check_max_rows:
        expanded_len = int(np.ptp(x0[time_col].agg([min, max])) / t_int)
        if expanded_len > MAX_ROWS:
            request_str = ('time expansion will lead to a dataframe with',
                           f'{expanded_len} rows. Continue (y/n)?')
            request = input(request_str)
            if request != 'y':
                raise ValueError('Refuse to expand large dataframe')
            else:
                MAX_ROWS = expanded_len

    x_expanded = pd.DataFrame(
        {time_col: pd.date_range(*x0[time_col].agg([min, max]).values,
                                 freq=pd.Timedelta(t_int))})

    x_expanded[time_col] = x_expanded[time_col].astype(x[time_col].dtype)

    # merge to original dataset to fill
    x0 = x0[[time_col, target]].merge(x_expanded, on=time_col, how='outer')
    for v in group:
        x0[v] = r[v]
    x0['is_expanded'] = x0[target].isnull()
    x0[target].fillna(False, inplace=True)

    x0 = x0[group + [time_col, target, 'is_expanded']].sort_values(time_col)
    return x0.set_index(time_col)


def _merge_nearby_events(x0, target, time_col, merge_interval, t_int):
    """
    Merge events closer than given threshold

    :param x0 : dataframe. expanded dataframe for given group
    :param target : str. Name of variable to use when labeling connected
    elements
    :param time_col : str. Sort dataset by this column
    :param t_int : timedelta. Median time difference between samples of the
    same group
    :param merge_interval : timedelta. Threshold to use when merging nearby
    events
    :return dataframe with nearby events merged
    """
    x_label = x0.loc[x0[target]].reset_index()
    x_label['dt_shift'] = x_label[time_col].shift(-1)
    x_label['dt_time'] = (x_label.dt_shift - x_label[time_col])

    x_merge = x_label[(x_label.dt_time <= merge_interval)
                      & (x_label.dt_time > t_int)]
    x0['target_final'] = x0[target]
    for _, r1 in x_merge.iterrows():
        x0.loc[str(r1[time_col]): str(r1.dt_shift), 'target_final'] = True
    return x0


def label_connected_elements_intervals(x, target, time_col, interval,
                                       group=None,
                                       label_col='labeled',
                                       time_unit='m',
                                       time_period=None,
                                       verbal=False,
                                       check_max_rows=True):
    """
    Label consecutive events (possibly by group) using function label from

    scipy.ndimage.measurements

    :param x : dataframe. main dataset with events to label.
    :param target : str. Name of variable to use when labeling connected
    elements
    :param time_col : str. Sort dataset by this column
    :param interval : float. Time threshold for samples part of the same event
    (in minutes). Default : 10.
    :param group : str. Name of grouping variable in dataset. If none, apply on
    entire
    dataset. Default : None
    :param label_col : str. Name to use for the labeled column.
    Default: 'labeled'
    :param time_unit: str. Name of time unit. Default: 'm' (minutes)
    :param time_period: str. Time period in pd.Timedelta string (e.g. '10S').
    Default: None (will be inferred from time column)
    :param verbal: bool. Whether to report verbally progress. Default: False
    :param check_max_rows: bool. Check number of rows after expansion and
    require user input if exceeded. Default: True
    :return array with labels of connected elements
    """
    # drop label column if already there
    x.drop(columns=label_col, errors='ignore', inplace=True)

    global MAX_ROWS
    MAX_ROWS = 1000000
    if group is None:
        group = ['dummy_group']
        x['dummy_group'] = True
    else:
        group = [group] if isinstance(group, str) else group

    # check whether timecol is in the index and reset it if so
    if time_col == x.index.name:
        x.reset_index(inplace=True)
        reset_idx = True
    else:
        reset_idx = False
    x.sort_values(group + [time_col], inplace=True)

    t_int = time_period or x.groupby(group)[time_col].diff().median()

    merge_interval = np.timedelta64(interval, time_unit)

    x_group = x[group].drop_duplicates()
    x.set_index(group, inplace=True)
    x.sort_index(inplace=True)
    x_labeled = []
    for _, r0 in progressbar(x_group.iterrows(),
                             total=len(x_group),
                             desc='Merging events',
                             verbal=verbal):
        x0 = _expand_dataframe(x, r0, group, time_col, target,
                               t_int, check_max_rows)

        x0 = _merge_nearby_events(x0, target, time_col, merge_interval, t_int)

        # label connected targets
        x0 = label_connected_elements(x0.reset_index(),
                                      'target_final',
                                      time_col,
                                      label_col=label_col)

        # remove extra rows that had been added to fill
        x0 = x0[x0.is_expanded.apply(lambda xx: xx is False)]

        x_labeled.append(x0)

    x_labeled = pd.concat(x_labeled)

    x = x.reset_index().merge(x_labeled[group + [time_col, label_col]],
                              on=group + [time_col],
                              how='left')

    x.drop(columns=['dummy_group', 'is_expanded'],
           inplace=True,
           errors='ignore')

    # re-set index if needed
    if reset_idx:
        x.set_index(time_col, inplace=True)

    return x


def summarize_connected_events(df, time_col, target_col, label_col,
                               group=None, time_unit='m',
                               agg_fun_extra=None,
                               extra_cols=None):
    """
    Summarize labeled events based on start, end, duration and ratio of

    actual labeled entries in each event. Further summarizing views are
    possible
    :param df: dataframe. Input dataframe with time and labeled column
    :param time_col: str. Name of time column
    :param target_col: str. Name of target col used in labeling process
    :param label_col: str. name of label column
    :param group: list/str. grouping columns. Default: [] (no grouping)
    :param time_unit: str. time unit for time features. Default: 'm' (minutes)
    :param agg_fun_extra: dict. Dictionary with columns as keys and functions
    to apply to those columns as values. Default: none (no extra functions)
    :param extra_cols: list/str. Name of columns resulting from extra
    aggregation functions. Default: None (will be inferred from agg_fun_extra)
    :returns dataframe with summary of each event
    """
    group = [] if group is None else group
    group = ([group] if isinstance(group, str) else group) + [label_col]

    def evt_duration(x):
        return (x.max() - x.min()) / np.timedelta64(1, time_unit)
    agg_fun = {time_col: [np.min, np.max, evt_duration],
               target_col: [np.sum, len]}

    # define extra functions for aggregations
    if agg_fun_extra is not None:
        agg_fun.update(agg_fun_extra)
        if extra_cols is None:
            extra_cols = []
            for k, v in agg_fun_extra.items():
                v_list = v if isinstance(v, (list, np.ndarray)) else [v]
                for k2 in np.arrange(len(v_list)):
                    extra_cols.append(f'{v}{k2}')
        else:
            extra_cols = (extra_cols
                          if isinstance(extra_cols, (list, np.ndarray))
                          else [extra_cols])
    else:
        extra_cols = []

    # summarize events with label greater than 0 and define column names
    evt = (df[df[label_col] > 0]
           .reset_index()
           .groupby(group)
           .agg(agg_fun))
    base_cols = ['start', 'end', f'duration_{time_unit}', 'is_outlier', 'ct']
    evt.columns = base_cols + extra_cols
    evt['rat_labeled'] = evt.is_outlier / evt.ct

    return evt.reset_index().drop(columns=['is_outlier', 'ct'])


def expand_dataframe(x, colnames=None, set_as_index=False):
    """
    Make all possible combinations of elements in input and put in dataframe

    :param x : list. List of elements (list, array) to combine
    :param colnames : list. List of column names. Should be the same size as x
    :param set_as_index : bool. Should the elements in colnames set as
    indices of the
    dataframe. Default : False
    :returns pandas dataframe with all possible combinations
    """
    ds = pd.DataFrame(list(itertools.product(*x)))
    if colnames is not None:
        ds.columns = colnames
    if set_as_index:
        ds.set_index(colnames, inplace=True)

    return ds


def subset_to_group(df, r, grouper, apply_subset=False):
    """
    Subset dataframe to multiple columns based on values in series

    :param df: pandas dataframe. Target dataframe to subset
    :param r: pandas series. Pandas series with values for subsetting
    :param grouper: list. List of items to use in subsetting
    :param apply_subset: bool. Whether to apply subsetting or simply return
    bool vector. Default: False
    :return: array with bool vector for subsetting or subsetted dataframe
    """
    subset_array = np.all(
            np.stack([df[g] == r[g] for g in grouper], axis=1),
            axis=1)
    if apply_subset:
        return df.loc[subset_array]
    else:
        return subset_array
