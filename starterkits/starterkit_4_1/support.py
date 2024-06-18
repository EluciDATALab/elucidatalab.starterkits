from sklearn.metrics import root_mean_squared_error, ConfusionMatrixDisplay, confusion_matrix
from starterkits import pipeline
import copy
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from functools import reduce
import zipfile


def get_data(DATA_PATH, force=False):
    """Exctract dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally.
    returns: subset engie dataset (2013-2016)
    """ 
    def extract_csv_file(fname):
        zip_fname = pipeline.download_from_repo(
            'SK_4_1',
            'IntelligentDataRetention.zip',
            str(DATA_PATH / 'SK_4_1' / 'IntelligentDataRetention.zip'),
            force=force)
        with zipfile.ZipFile(zip_fname, 'r') as zip:
            zip.extract('IntelligentDataRetention.csv', path=str(DATA_PATH / 'SK_4_1'))
        os.remove(zip_fname)
        return fname
    fname = (pipeline.File(extract_csv_file, 
                           str(DATA_PATH /'SK_4_1' / 'IntelligentDataRetention.csv'))
             .make(force=force))

    df = pd.read_csv(fname, parse_dates=[0])

    return df


def load_data(DATA_PATH, force=False):
    """Read in data as dataframe and do some preprocessing."""

    def get_drives(x, gap_size=5):
        return ((x.index.to_series().diff()) > gap_size).cumsum()

    def small_drive(x, min_duration=10):
        # require more than 10 seconds of data
        return len(x) > min_duration

    df = get_data(DATA_PATH, force=force)

    df["time"] = df.time.view(np.int64) // 10 ** 9  # convert to epoch time
    df = df.set_index("time")

    df["drive"] = get_drives(df)
    df = df.groupby("drive").filter(small_drive)  # remove small drives

    return df


def define_slope(point1, point2):
    """Calculate the slope between two points."""
    # todo: fix divide by zero:
    #       what should slope return if point2[0] == point1[0] ?

    if point2[0] != point1[0]:
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        slope = 0

    return slope


def new_window(CD, c, d):
    """Calculate the new window for trend area."""
    u = (c[0], c[1] + CD)
    l = (c[0], c[1] - CD)
    upper_max = define_slope(u, d)
    lower_min = define_slope(l, d)
    return upper_max, lower_min


def SDT(t, v, d, upper_max, lower_min, CD):
    """Apply the Swinging Door Trending (SDT) algorithm on point (t,v).

    Given the boundaries, upper_max and lower_min, and the previous point to
    compress, the SDT algorithm will compute the new boundaries and will check
    if the current point (t,v) is in the compression area.
    
    Parameters
    ----------
    t : float
        The first dimension of the point (usually the timestep)
    v : float
        The second dimension of the point
    d : (float, float)
        The previous point
    upper_max : float
        The current upper boundary
    lower_min : float
        The current lower boundary
    CD : float
        The compression deviation
        
    Returns
    -------
    c : (float, float)
        New point
    d : (float, float)
        Previous point
    upper_max : float
        New upper boundary
    lower_min : float
        New_lower boundary
    """
    p = copy.copy(d)
    c = (np.nan, np.nan)
    u = (d[0], d[1] + CD)
    l = (d[0], d[1] - CD)
    d = (t, v)
    s_upper = define_slope(u, d)
    s_lower = define_slope(l, d)
    if s_upper > upper_max:
        upper_max = s_upper
        if upper_max > lower_min:
            s_o = define_slope(p, d)
            c_u = (u[1] - p[1] + s_o * p[0] - lower_min * u[0]) / (s_o - lower_min)
            c = (c_u, (u[1] + lower_min * (c_u - u[0]) - CD / 2))
            upper_max, lower_min = new_window(CD, c, d)
            c = p
            
    if s_lower < lower_min:
        lower_min = s_lower
        if upper_max > lower_min:
            s_o = define_slope(p, d)
            c_l = (l[1] - p[1] + s_o * p[0] - upper_max * l[0]) / (s_o - upper_max)
            c = (c_l, (l[1] + upper_max * (c_l - l[0]) - CD / 2))
            upper_max, lower_min = new_window(CD, c, d)
            c = p
            
    return c, d, upper_max, lower_min


def calc_compression_deviation(mode, data):
    """Calculate the compression deviation (CD) value for a given dataset.

    This is an automatic way of approximating good CD values.
    
    Parameters
    ----------
    mode : ['MEAN', 'ZMEAN', 'RANGE']
        The mode to use for the calculation of the CD value
    data : ndarray
        The data used to calculate the CD on
        
    Returns
    -------
    CD : float
        Compression delta
    """
    slopes = [np.abs(define_slope(x, y)) for x, y in zip(data, data[1:])]
    if mode == 'MEAN':
        CD = np.mean(slopes)
    elif mode == 'ZMEAN':
        CD = np.mean(slopes[slopes != 0])
    elif mode == 'RANGE':
        CD = (np.max(slopes) + np.min(slopes)) / 2
    else:
        raise ValueError(f"Unknown mode {mode}")
    return CD


def SSDT(data, mode='ZMEAN'):
    """Apply the SSDT algorithm on the given data.

    Apply the Self-definition Swinging Door Trending (SSDT) algorithm on the
    given data. This method will first calculate a compression deviation (CD)
    parameter based on the mode and will compress the data afterward by
    sequentially applying the SDT algorithm on all points.

    SSDT = SDT + calc_compression_deviation

    Parameters
    ----------
    data : ndarray
        The dataset that needs to be compressed
    mode : ['MEAN', 'ZMEAN', 'RANGE']
        The mode that is chosen to calculate the CD parameter
        
    Returns
    -------
    compressed_data : ndarray
        The compressed version of data
    """
    compressed_data = []
    start = data[0]
    d = c = start
    CD = calc_compression_deviation(mode, data)
    upper_max = -np.Inf
    lower_min = np.Inf
    compressed_data.append(c)
    for row in data[1:]:
        c, d, upper_max, lower_min = SDT(row[0], row[1], d, upper_max, lower_min, CD)
        compressed_data.append(c)
 
    return compressed_data


def SSDT_multi(data, mode='ZMEAN'):
    """Applying the SSDT algorithm on the given multi-dimensional data.

    Applying the Self-definition Swinging Door Trending (SSDT) algorithm on the
    given multi-dimensional data. The data consits of at least two dimensions
    where the first is always the timestep. This method will first calculate a
    compression deviation (CD) parameter for each dimension based on the
    mode and will compress each dimension sequentially.

    SSDT = SDT + calc_compression_deviation

    Parameters
    ----------
    data : ndarray
        The dataset that needs to be compressed
    mode : ['MEAN', 'ZMEAN', 'RANGE']
        The mode that is chosen to calculate the CD parameter
        
    Returns
    -------
    compressed_data: ndarray
        The compressed version of data
    """
    compressed_data = []
    start = data[0]
    d = c = start
    CDs = []
    for col in range(1, len(start)):
        CDs.append(calc_compression_deviation(mode, data))
    size_signals = len(start) - 1
    upper_max = [-np.Inf] * size_signals
    lower_min = [np.Inf] * size_signals
    compressed_data.append(c)
    old_d = start
    for row in data[1:]:
        t = row[0]
        new_c = []
        new_d = []
        new_upper = []
        new_lower = []
        for i, v in enumerate(row[1:]):  # ignore time(first) value
            d = (old_d[0], old_d[i + 1])
            c, d, upper, lower = SDT(t, v, d, upper_max[i], lower_min[i], CDs[i])
            new_c.append(c[1])
            new_d.append(d[1])
            new_upper.append(upper)
            new_lower.append(lower)
        if ~np.isnan(new_c).all():  # some points cannot be compressed, save points
            new_c = list(row[1:])
        compressed_data.append([t] + new_c)
        old_d = [t] + new_d
        upper_max = new_upper
        lower_min = new_lower
    return compressed_data


def compress_df_SSDT(df, group, mode='ZMEAN'):
    """Apply SSDT to the data in `df`, grouped by `group`."""
    df_return = []
    for name, df_sub in tqdm(df.groupby(group), total=df.drive.nunique()):
        data = df_sub.drop(columns=[group]).values
        compress_data = SSDT(data, mode=mode)
        df_temp = pd.DataFrame(compress_data, columns=['time', 'signal']).dropna().copy()
        df_temp[group] = name
        df_return.append(df_temp)
    df_return = pd.concat(df_return)
    df_return['time'] = np.round(df_return.time).astype(int)
    return df_return


def separate_compression(df):
    """ Consider each signal as separate time series and compress"""
    signals = ['maxXAcc', 'maxYAcc', 'maxZAcc', 'minXAcc', 'minYAcc', 'minZAcc']

    dfs = [compress_df_SSDT(df.reset_index()[['time', sig, 'drive']], 'drive').rename(columns={'signal': sig})
           for sig in signals]

    df_compressed_individually = reduce(lambda left, right: pd.merge(left, right, on=['time', 'drive'], how='outer'),
                                        dfs).set_index('time')

    df_compressed_individually = df_compressed_individually[~df_compressed_individually.index.duplicated()]

    return df_compressed_individually


def compress_df_SSDT_multi(df, signals, group, mode='ZMEAN'):
    """Apply SSDT to multi-dimensional data in `df`, grouped by `group`."""
    df_return = []
    for name, df_sub in tqdm(df.groupby(group), total=df.drive.nunique()):
        data = df_sub[['time'] + signals].values
        compress_data = SSDT_multi(data, mode)
        df_temp = pd.DataFrame(compress_data, columns=['time'] + signals).dropna().copy()
        df_temp[group] = name
        df_return.append(df_temp)
    df_return = pd.concat(df_return)
    df_return['time'] = np.round(df_return.time).astype(int)
    #  df_return.set_index('time', inplace=True)
    return df_return


def combined_compression(df):
    """ Compress signals simultaneously"""
    df_comb_min = df.reset_index()[['time', 'minXAcc', 'minYAcc', 'minZAcc', 'drive']]
    df_comb_max = df.reset_index()[['time', 'maxXAcc', 'maxYAcc', 'maxZAcc', 'drive']]

    df_comb_min_compressed = compress_df_SSDT_multi(df_comb_min, ['minXAcc', 'minYAcc', 'minZAcc'], 'drive')
    df_comb_max_compressed = compress_df_SSDT_multi(df_comb_max, ['maxXAcc', 'maxYAcc', 'maxZAcc'], 'drive')

    df_compressed_combined = df_comb_min_compressed.merge(df_comb_max_compressed, on=['time', 'drive'], how='outer')
    df_compressed_combined = df_compressed_combined.set_index('time').sort_index()

    return df_compressed_combined


def compression_ratio(original, compressed_ind, compress_comb):
    """ Determine amount of compression by considering number of datapoints"""
    length = len(original)
    length_compression_individually = len(compressed_ind)
    length_compression_combined = len(compress_comb)

    print(f'Original size: {length}')
    print(f'Normal compression: {length_compression_individually}')
    print(f'Combined compression: {length_compression_combined}')
    print(f'Compression ratio normal compression: {length / length_compression_individually:.3f}')
    print(f'Compression ratio combined compression: {length / length_compression_combined:.3f}')


def file_size(original, compress_indiv, compressed_comb):
    """ Determine amount of compression by considering csv file size"""
    original.astype(np.float32).to_csv('OrginalData.csv')
    compress_indiv.astype(np.float32).to_csv('FirstCompressed.csv')
    compressed_comb.astype(np.float32).to_csv('SecondCompressed.csv')

    original_size = os.stat('OrginalData.csv').st_size
    compr1_size = os.stat('FirstCompressed.csv').st_size
    compr2_size = os.stat('SecondCompressed.csv').st_size
    print(f'The file size of the original data is {original_size / 1e6:.2f} MB')
    print(
        f'The file size of the first compressed data is {compr1_size / 1e6:.2f} '
        f'MB which is {original_size / compr1_size:.2f} times smaller')
    print(
        f'The file size of the second compressed data is {compr2_size / 1e6:.2f} '
        f'MB which is {original_size / compr2_size:.2f} times smaller')


def rmse(original, compress_indiv, compressed_comb):
    """ Determine amount of compression by considering rmse of reconstructed signals"""
    df_reconstructed_individually = compress_indiv.reindex(original.index).interpolate(method='linear')
    df_reconstructed_combined = compressed_comb.reindex(original.index).interpolate(method='linear')

    rmse_individually = root_mean_squared_error(original.drop(columns=['drive']),
                                           df_reconstructed_individually.drop(columns=['drive']))
    rmse_combined = root_mean_squared_error(original.drop(columns=['drive']),
                                       df_reconstructed_combined.drop(columns=['drive']))

    print(f'RMSE for the sequential application of SDT: {rmse_individually:.2}')
    print(f'RMSE for the combined application of SDT: {rmse_combined:.2}')


def add_percentage_above_threshold(df, event_thresholds):
    for axis, thresholds in event_thresholds.items():
        for name, limits in thresholds.items():
            df[f'{name + axis}_thre_perc'] = 100 * df[f'{name + axis}'] / limits
    return df


def add_event_type(df, event_tree):
    cols = ['minXAcc_thre_perc',
            'maxXAcc_thre_perc',
            'minYAcc_thre_perc',
            'maxYAcc_thre_perc',
            'minZAcc_thre_perc',
            'maxZAcc_thre_perc']
    for event, thresholds in event_tree.items():
        if thresholds is None:
            continue
        low, high = thresholds
        mask = ((low < df[cols]).any(axis=1)) & ((df[cols] < high).any(axis=1))
        df.loc[mask, 'event'] = event
    return df


def flag_events(df, event_thresholds, event_tree):
    """Flag data points above/below given thresholds as events and
    add event type."""
    df = df.copy(deep=True)
    df = add_percentage_above_threshold(df, event_thresholds)
    df = add_event_type(df, event_tree)
    event_idx = df.loc[df.event.notnull()].index
    detections = df.loc[event_idx].event
    return detections


def event_detection(original, compress_indiv, compressed_comb):
    """ Perform event detection on all datasets"""
    event_thresholds = {
        'XAcc': {'min': -2.5, 'max': 1.5},
        'YAcc': {'min': -1.8, 'max': 1.3},
        'ZAcc': {'min': -2.5, 'max': 2},
    }

    event_tree = {
        "No event": None,
        "Warning": [100, 150],
        "Incident": [150, 200],
        "Hard_incident": [200, np.Inf],
    }

    event_labels = sorted(event_tree)

    y_original = flag_events(original, event_thresholds, event_tree)
    y_compressed_individually = flag_events(compress_indiv, event_thresholds, event_tree)
    y_compressed_combined = flag_events(compressed_comb, event_thresholds, event_tree)

    return y_original, y_compressed_individually, y_compressed_combined, event_labels
