# ©, 2022, Sirris
# owner: HCAB

import os
import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL

from starterkits import pipeline, DATA_PATH
from starterkits.preprocessing.outliers import boxplot_outliers
from starterkits.utils.df_utils import label_connected_elements_intervals

import zipfile
import ipywidgets as widgets
import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)



def get_data(DATA_PATH, force=False):
    """Extract dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally.
    returns: subset engie dataset (2013-2016)
    """ 
    def extract_csv_file(fname):
        target_folder = DATA_PATH / 'SK_3_4'
        zip_fname = pipeline.download_from_repo(
            'SK_3_4',
            'la-haute-borne-data-2013-2016.zip',
            str(target_folder / 'la-haute-borne-data-2013-2016.zip'),
            force=force)

        with zipfile.ZipFile(zip_fname, 'r') as zip:
            zip.extract('la-haute-borne-data-2013-2016.csv', path=str(target_folder))
        os.remove(zip_fname)
        return fname
    fname = (pipeline.File(extract_csv_file, 
                           str(DATA_PATH / 'SK_3_4' / 'la-haute-borne-data-2013-2016.csv'))
             .make(force=force))

    df = pd.read_csv(fname, sep=';')

    return df


def load_data(DATA_PATH, force=False):
    """Reads in data and does some preprocessing. The function also creates daily and turbine-specific datasets

    dataset can be downloaded from
    # https://opendata-renewables.engie.com/explore/dataset/d543716b-368d-4c53-8fb1-55addbe8d3ad/information#
    :returns ds. dataframe containing the entire dataset
    :returns dst. dataframe with data for a single turbine
    :returns df_missing. dataframe with fake missing data for imputation section
    :returns missing_events. dataframe with summary of missing data events
    :returns missing_events_sum. dataframe with summary of missing data events per duration
    """

    ds = get_data(DATA_PATH, force=force)
    ds.index = pd.DatetimeIndex(
        pd.to_datetime(ds.Date_time.apply(lambda x: x.split('+')[0])),
        name='datetime')
    renamer = {'Wind_turbine_name': 'turbine',
                'P_avg': 'power',
                'Ws_avg': 'wind_speed',
                'Ot_avg': 'temperature',
                'Wa_avg': 'wind_direction'}
    ds.rename(columns=renamer, inplace=True)
    ds = ds[list(renamer.values())]

    ds.dropna(inplace=True)
    ds.sort_index(inplace=True)

    turbine = 'R80721'
    dst = ds[ds.turbine == turbine].drop(columns='turbine')
    
    # get fake missing data events for imputation section
    df_missing, missing_events, missing_events_sum = make_random_missing(ds)

    col_labels = {'power': 'Power (kW)', 
                  'wind_speed': 'Wind speed (m/s)', 
                  'wind_direction': 'Wind direction (deg)', 
                  'temperature': 'Temperature (Celsius)'}
    
    return ds, dst, df_missing, missing_events, missing_events_sum, col_labels


def define_slider_date_range(start_date, end_date, date_format='%d/%m/%Y'):
    """Define slider range with custom start, end and freq

    Args:
        start_date: datetime object. Starting date of slider
        end_date: datetime object. Ending date of slider
        date_format: str. format to use for printing dates. Default: '%d/%m/%Y'

    Returns:
        widget with date slider
    """


    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(date_format), date) for date in dates]
    index = (0, len(options)-1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '400px'},
        continuous_update=False
    )

    return selection_range_slider


def stl_decomposition(df, period=365):
    """Perform seasonal trend decomposition and return the 4 underlying signals

    Args:
        df: series. Attribute to be decomposed with datetime index
        period: int. Period to use in decomposition. Default: 365

    Returns:
        df_stl: dataframe. Dataframe containing the 4 underlying signals: observed, trend, seasonal and residual
    """
    decomp = STL(df, period=period).fit()

    df_stl = pd.DataFrame({'observed': decomp.observed,
                           'trend': decomp.trend,
                           'seasonal': decomp.seasonal,
                           'residual': decomp.resid},
                          index=df.index)

    return df_stl


def summarize_outliers(df, n_iqr, int_minutes):
    """Summarize detected outliers by first joining them into outlier events

    Args:
        df: dataframe. input dataset
        n_iqr: int. Number of IQR from the lower and upper quartiles
        int_minutes: int. Number of minutes to use when merging nearby events

    Returns:

    """
    df['outliers'] = boxplot_outliers(df.residual, n_iqr)[0]
    if df['outliers'].sum() == 0:
        return None, None
    else:
        # label connected outlier events
        df = label_connected_elements_intervals(df.reset_index(),
                                                'outliers',
                                                'datetime',
                                                interval=int_minutes,
                                                label_col='label_outlier')

        # summarize outlier events
        outlier_events = (df[df.label_outlier > 0]
                          .groupby('label_outlier')
                          .agg({'datetime': [min,
                                             max,
                                             lambda x: (x.max() - x.min()) / np.timedelta64(1, 'm')],
                                'temperature': [min, max, np.median],
                                'residual': [min, max, np.median]}))
        outlier_events.columns = ['Start', 'End', 'Duration (minutes)',
                                  'Temperature min', 'Temperature max', 'Temperature median',
                                  'Residual min', 'Residual max', 'Residual median']
        outlier_events['Start'] = outlier_events['Start'].dt.strftime('%Y-%m-%d %H:%M')
        outlier_events['End'] = outlier_events['End'].dt.strftime('%Y-%m-%d %H:%M')
        outlier_events['Event ID'] = np.arange(len(outlier_events))
        for v in ['Temperature min', 'Temperature max', 'Temperature median',
                  'Residual min', 'Residual max', 'Residual median']:
            outlier_events[v] = np.round(outlier_events[v], 1)
        evt_id = outlier_events['Event ID']
        outlier_events.drop(columns='Event ID', inplace=True)
        outlier_events.insert(0, 'Event ID', evt_id)
        df.set_index('datetime', inplace=True)

        return outlier_events, df


def get_fleet_outliers(df):
    """

    Args:
        df: dataframe. Input dataframe

    Returns:
        outlier_fleet_evt: dataframe. Outlier events summary
    """
    # count number of remaining turbines with non-zero value at each time point per turbine
    def nzero(x):
        x = pd.pivot_table(x, index='datetime', columns='turbine', values='power')
        n_zero = x.copy()
        for t in x.columns:
            n_zero[t] = (x.drop(columns=t) > 0).sum(axis=1)

        return (pd.melt(n_zero.reset_index(),
                        id_vars='datetime',
                        var_name='turbine',
                        value_name='n_zero'))

    v = 'power'
    df['fleet_value'] = df.groupby('datetime')[v].median()
    df['residual_fleet'] = df[v] - df.fleet_value
    # label outliers
    df['outlier_fleet'] = boxplot_outliers(df.residual_fleet.abs(), 5)[0]

    # keep only as outliers those where the remaining 3 turbines have non-zero power value
    nz = nzero(df)
    df = df.reset_index().merge(nz, on=['datetime', 'turbine'])
    df['outlier_fleet'] = (df['outlier_fleet']) & (df['n_zero'] == 3)

    # label connected elements. outliers less than 20min apart are merged together
    df = (label_connected_elements_intervals(df.reset_index(),
                                             'outlier_fleet',
                                             'datetime',
                                             20,
                                             group='turbine',
                                             label_col='outlier_evt_fleet')
          .set_index('datetime'))

    # summarize outlier events
    outlier_fleet_evt = (df[df.outlier_evt_fleet > 0]
                         .reset_index()
                         .groupby(['turbine', 'outlier_evt_fleet'])
                         ['datetime']
                         .agg([min, max]))
    outlier_fleet_evt.columns = ['Start', 'End']
    outlier_fleet_evt['Duration_min'] = (outlier_fleet_evt.End - outlier_fleet_evt.Start) / np.timedelta64(1, 'm')
    outlier_fleet_evt.reset_index(inplace=True)

    outlier_fleet_evt['Event ID'] = (outlier_fleet_evt.turbine
                                     + '_'
                                     + outlier_fleet_evt.Start.astype(str))

    outlier_fleet_evt = (outlier_fleet_evt
    [outlier_fleet_evt
            .Duration_min
            .between(30, 120)])

    df.drop(columns=['fleet_value', 'residual_fleet',
                     'outlier_fleet', 'outlier_evt_fleet'],
            inplace=True)

    return outlier_fleet_evt


def make_random_missing(df):
    """Create random missing events with a range of durations

    Args:
        df: dataframe. input dataset

    Returns:
        df_missing: dataframe. new dataset with missing values
        missing_events: dataframe. summary of missing data events
        missing_events_sum: dataframe. summary of events per duration
    """

    # define range of durations for missing data events
    durations = np.arange(10, 240, 10)
    prob = np.logspace(0.9, 0.1, len(durations))
    prob = prob / prob.sum()

    df_missing = df[['turbine', 'wind_speed']].copy()
    df_missing['wind_speed_original'] = df_missing.wind_speed.copy()
    df_missing['missing_event'] = 0

    # create 100 random events
    for i in np.arange(100):
        duration = int(np.random.choice(durations, size=1, p=prob)[0])
        r = df[df.wind_speed.notnull()].sample(1).iloc[0]
        en = r.name + np.timedelta64(duration, 'm')
        id = (df.index >= r.name) & (df.index < en) & (df.turbine == r.turbine)
        df_missing.loc[id, 'wind_speed'] = np.nan
        df_missing.loc[id, 'missing_event'] = df_missing.missing_event.max() + 1

    # make summary of events
    missing_events = (df_missing[df_missing.missing_event > 0]
                      .reset_index()
                      .groupby(['turbine','missing_event'])
                      .datetime
                      .agg([lambda x: np.ptp(x) / np.timedelta64(1, 'm'),
                            np.min, np.max]))
    missing_events.columns = ['Duration (min)', 'Start', 'End']
    missing_events['Duration (min)'] = missing_events['Duration (min)']
    missing_events.reset_index(inplace=True)

    # summarize per duration
    missing_events_sum = missing_events.groupby('Duration (min)').size().reset_index(name='# Events')

    missing_events['event_id'] = (missing_events.turbine
                                  + '_' +
                                  missing_events.Start.astype(str))

    return df_missing, missing_events, missing_events_sum
