# ©, 2022, Sirris
# owner: HCAB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays

from datetime import datetime, timedelta
from IPython.display import Markdown, display
import zipfile
import os

import multiprocessing
from joblib import Parallel, delayed

from starterkits import pipeline, DATA_PATH

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from starterkits.visualization import vis_plotly as vp
from starterkits.visualization import vis_plotly_tools as vpt
from starterkits.visualization import vis_plotly_widgets as vpw

from ipywidgets import interact, interact_manual, interactive, fixed
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

target = 'consumption'
features = ['Pastday', 'Pastweek', 'Hour', 'Weekday', 'Month', 'Holiday', 'Temperature']


def read_consumption_data(force=False):
    """Extract and return dataset from raw data file in server.

    This dataset will be saved locally.
    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

    returns: Household power consumption dataset.
    """

    # create data folder if it does not exist locally
    target_folder = DATA_PATH / 'SK_1_3'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    def extract_txt_file(fname):
        zip_fname = pipeline.download_from_repo(
            'SK_1_3',
            'household_power_consumption.txt.zip',
            str(target_folder / 'household_power_consumption.txt.zip'),
            force=force)

        with zipfile.ZipFile(zip_fname, 'r') as zip:
            zip.extract('household_power_consumption.txt', path=str(fname.parent))
        os.remove(zip_fname)
        return fname

    fname = (pipeline.File(extract_txt_file, 
                           str(DATA_PATH / 'SK_1_3' / 'household_power_consumption.txt'))
             .make(force=force))

    parse = lambda x, y: datetime.strptime(x+' '+y, '%d/%m/%Y %H:%M:%S')
    df = pd.read_csv(fname, sep=';', na_values=['?'], index_col=0,
                     parse_dates=[['Date', 'Time']], date_parser=parse)
    df = df.loc['2007-01-01':]
    return df


def read_climate_data_(fname):
    """ Reads climate data directly from web. Use `read_climate_data` if possible to download from private server. """
    def read_temperature(y):
        ds = pd.read_csv(f'https://www.ncei.noaa.gov/data/global-hourly/access/{y}/07156099999.csv')
        ds['temperature'] = ds.TMP.apply(lambda x: x.replace(',', '.')).astype(float) / 10
        ds.index = pd.DatetimeIndex(ds.DATE, name='datetime')
        return ds[['temperature']]
    temperature = pd.concat([read_temperature(y) 
                             for y in np.arange(2007, 2011)])
    temperature.reset_index().to_csv(fname, index=False)
    return fname


def read_climate_data(force=False):
    """ Read climate data.

    Source: 'https://www.ncei.noaa.gov/data/global-hourly/access/{YYYY}/07156099999.csv', where {YYYY} = 2007-2011
    The preprocessed dataset can be created directly from the web by running `read_climate_data_()`.
    """

    # create data folder if it does not exist locally
    target_folder = DATA_PATH / 'SK_1_3'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    def extract_csv_file(fname):
        zip_fname = pipeline.download_from_repo(
            'SK_1_3',
            'hourly_temperature.csv.zip',
            str(target_folder / 'hourly_temperature.csv.zip'),
            force=force)

        with zipfile.ZipFile(zip_fname, 'r') as zip:
            zip.extract('hourly_temperature.csv', path=str(fname.parent))
        os.remove(zip_fname)
        return fname

    fname = (pipeline.File(extract_csv_file,
                           str(DATA_PATH / 'SK_1_3' / 'hourly_temperature.csv'))
             .make(force=force))

    temperature = pd.read_csv(fname).set_index('datetime')
    temperature.index = pd.to_datetime(temperature.index)
    return temperature.temperature


def printmd(string):
    display(Markdown(string))


def get_months(feats):
    start = feats.index[0]
    end = feats.index[-1]

    daterange = pd.date_range(start, end,freq='M')
    months = [(d.year,d.month) for d in pd.date_range(start, end,freq='M')]
    return months


def predictions_window(model, feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):

    months = get_months(feats)

    start_month_train = months[start_month_train_idx]
    end_month_train = months[start_month_train_idx + window_size_train]

    start_month_test = months[start_month_train_idx + window_size_train + offset_test]
    end_month_test = months[start_month_train_idx + window_size_train + offset_test + window_size_test]

    # Add time specifier in the end date, otherwise, the full first day of the end month is included
    # Then, exclude the final element so that the data ends at 23h the day before
    train_data = feats["{}-{}-1".format(*start_month_train):"{}-{}-1 00:00:00".format(*end_month_train)][:-1]
    test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]

    model.fit(train_data[features], train_data[target])
    test_predict = model.predict(test_data[features])

    return test_predict


def score_window(model, feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):
    """
    Get the mean absolute error of a model trained a certain time window and evaluated on some month(s)
    after that window.

    model: instance of an sklearn regressor
    start_month_train_idx: index of the starting month of the training window
    window_size_train: number of months in the training window
    offset_test: how many months between the last month in the training window and the first month
        of the test window. Default: 1.
    window_size_test: number of months in the testing window
    """
    months = get_months(feats)

    start_month_train = months[start_month_train_idx]
    end_month_train = months[start_month_train_idx + window_size_train]

    start_month_test = months[start_month_train_idx + window_size_train + offset_test]
    end_month_test = months[start_month_train_idx + window_size_train + offset_test + window_size_test]

    # Add time specifier in the end date, otherwise, the full first day of the end month is included
    # Then, exclude the final element so that the data ends at 23h the day before
    train_data = feats["{}-{}-1".format(*start_month_train):"{}-{}-1 00:00:00".format(*end_month_train)][:-1]
    test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]

    model.fit(train_data[features], train_data[target])
    test_predict = model.predict(test_data[features])
    return mean_absolute_error(test_data[target], test_predict)


def get_true_values(feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):

    #months = get_months(feats)

    #start_month_test = months[start_month_train_idx + window_size_train + offset_test + 1]
    #end_month_test = months[start_month_train_idx + window_size_train + offset_test + 1 + window_size_test]

    #test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]
    test_idxes = get_datetimes(feats, start_month_train_idx, window_size_train,
                               offset_test=offset_test, window_size_test=window_size_test)
    test_data = feats.loc[test_idxes]
    return test_data[target]


def get_datetimes(feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):

    months = get_months(feats)

    start_month_train = months[start_month_train_idx]
    end_month_train = months[start_month_train_idx + window_size_train]

    start_month_test = months[start_month_train_idx + window_size_train + offset_test]
    end_month_test = months[start_month_train_idx + window_size_train + offset_test + window_size_test]

    # Add time specifier in the end date, otherwise, the full first day of the end month is included
    # Then, exclude the final element so that the data ends at 23h the day before

    test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]

    return test_data.index


def parallel_evaluation(model, feats, window_size_train, offset_test=1, window_size_test=1):
    num_cores = multiprocessing.cpu_count()
    months = get_months(feats)
    if window_size_train == 'fullperiod':
        max_window_size = len(months) - offset_test - 1 - window_size_test
        scores = Parallel(n_jobs=num_cores)(delayed(score_window)(model, feats, 0, w, offset_test, window_size_test)
                                                for w in range(1, max_window_size))
    else:
        max_idx_month = len(months) - offset_test - 1 - window_size_test - window_size_train
        scores = Parallel(n_jobs=num_cores)(delayed(score_window)(model, feats, m, window_size_train, offset_test, window_size_test)
                                                for m in range(0, max_idx_month))

    return scores


