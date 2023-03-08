# ©, 2022, Sirris
# owner: HCAB

import numpy as np
import pandas as pd

import warnings
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets

from starterkits.statistical_tools.angular import circular_mean
from starterkits.visualization import vis_plotly as vp
from starterkits.visualization import vis_tools
from starterkits.visualization import vis_plotly_tools as vpt
from starterkits.visualization import vis_plotly_widgets as vpw

from starterkits.starterkit_3_4.support import *


def table_data_overview(df):
    """Print overview of data with aggregates per turbine

    Args:
        df: dataframe. Main dataset

    Returns:

    """
    # subset the data for speed purposes
    df = df[df.temperature > -20]
    # df = pd.concat([df.iloc[:100,:], df.iloc[100:].sample(50000, replace=False)])
    df = df.loc['2016':]

    # function that defines the settings for the range slider for a given attribute
    def get_range(col):
        val = df[col]
        step = np.median(np.diff(np.linspace(*np.nanpercentile(val, [1, 99]), 100)))
        window = tuple(np.nanpercentile(val, [25, 75]))
        mi = val.min()
        ma = val.max() + step
        return (mi, ma, window, step)

    # DEFINE INTERACTIBLES
    columns=['temperature', 'wind_speed','wind_direction', 'power']
    col = widgets.Dropdown(options=[(vis_tools.prettify(c), c) for c in columns],
                           value='power',
                           description='Attribute')

    s_range = get_range(col.value)
    slider = widgets.FloatRangeSlider(continuous_update=False,
                                      min=s_range[0], max=s_range[1],
                                      value=s_range[2], step=s_range[3],
                                      description='Value range')

    # function to update slider range on change of attribute
    def on_column_change(args):
        s_range = get_range(col.value)
        slider.min = s_range[0]
        slider.max = s_range[1]
        slider.value = s_range[2]
        slider.step = s_range[3]
    col.observe(on_column_change, names='value')

    # MAIN PLOTTING FUNCTION
    def subset_table(col_name, window):
        num_cols = ['power', 'temperature', 'wind_speed', 'wind_direction']
        # limit table to range
        table = df.loc[(df[col_name].between(window[0], window[1])), ['turbine'] + num_cols]

        # get average aggregate per turbine
        table_sum = (table
                     .groupby('turbine')
                     .agg({'power': 'mean',
                           'temperature': 'mean',
                           'wind_speed': 'mean',
                           'wind_direction': circular_mean})
                     .reset_index())

        # round values in table
        for d in [table, table_sum]:
            for c in num_cols:
                d[c] = np.round(d[c], 2)

        # final table processing
        table.reset_index(inplace=True)
        table['datetime'] = table['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        table_sum.insert(0, 'datetime', 'AVERAGE')
        # set bold text
        for c in table_sum:
            table_sum[c] = table_sum[c].astype(str).apply(lambda x: f'<b>{x}</b>')
        table = table.sort_values(['datetime', 'turbine'])#.head(100)
        empty = pd.DataFrame(np.repeat('', table.shape[1])
                             .reshape((1, table.shape[1])), 
                             columns=table.columns, 
                             index=[0])
        table_concat = pd.concat([table_sum, empty, table], axis=0)
        table_concat.columns = [f'<b>{x}</b>' 
                                for x in vis_tools.prettify(table_concat.columns)]
        
        fig = go.Figure(go.Table(header={'values': list(table_concat.columns)},
                                 cells={'values': [table_concat[c] for c in table_concat.columns]}))
        fig.show(renderer="colab")
        # vt.plot_table(table_concat, page_size=20)

    return subset_table, col, slider


def plot_resampling(df, col_labels):
    """Render interactive pannel allowing to test different resampling settings

    Args:
        df: dataframe. Main dataset
        col_labels: dict. Dictionary with labels per column

    Returns:

    """
    # DEFINE WIDGETS
    unit = widgets.Dropdown(options=[('Hour', 'H'),
                                     ('Day', 'D'),
                                     ('Week', 'W'),
                                     ('Month', 'M')],
                            description='Time unit',
                            value='D')
    slider = widgets.IntSlider(value=3, min=1, max=30, step=1,
                               description='# time units', 
                               continuous_update=False)

    aggfun = widgets.Dropdown(options=[('Mean', np.mean), 
                                       ('Median', np.median), 
                                       ('Sum', np.sum)],
                              value=np.mean,
                              description='Aggregation function',
                              style={'description_width': 'initial'})

    # DEFINE PLOTTING FUNCTION
    def resample_plots(u, c, a):
        # check if any resampling is needed (cycles > 0)
        if c > 0:
            freq = '%d%s' % (c, u)
            df_resample = df.resample(freq).agg(a)
        else:
            df_resample = df
        columns = ['power', 'wind_speed','temperature']
        fig = vp.plot_multiple_time_series(
            [df_resample[c] for c in columns],
            series_names=vis_tools.prettify(columns), 
            same_plot=False, 
            show_legend=False)
            
        vpt.update_labels_subplots(fig, {'yaxis_title': col_labels}, columns)
        fig.show(renderer="colab")

    return resample_plots, unit, slider, aggfun


def plot_smoothing(df):
    """Render interactive panel to test different smoothing algorithms

    Args:
        df: dataframe. Main dataset

    Returns:

    """

    # function to get time period
    def get_time_period(x, time_unit):
        return np.median(np.diff(x.index)) / np.timedelta64(1, time_unit)
    # DEFINE METHODS ------------
    def rolling(x, c, window, time_unit):
        "Rolling window smoothing"
        x = x[c]
        if window:
            return x.rolling(f'{window}{time_unit}').median()
        else:
            return x    

    def savgol(x, c, window, time_unit):
        "Savgol filter smoothing"
        x = x[c]
        if window:
            window = window / get_time_period(x, time_unit)
            window = len(x)-2 if window >= len(x) else window
            window = int(window + 1 if window % 2 == 0 else window)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pd.Series(savgol_filter(x, window, 2), index=x.index, name=x.name)
        else:
            return x
    def gaussian(x, c, window, time_unit):
        "Gaussian smoothing"
        x = x[c]
        if window:
            window = window / get_time_period(x, time_unit)
            # width of filter accoring to scipy dos: width = 2*int(4*sigma + 0.5) + 1
            sigma = ((window - 1) / 2 - 0.5) / 4
            return pd.Series(gaussian_filter1d(x, sigma), index=x.index, name=x.name)
        else:
            return x
        
    methods = {
        'Rolling median': rolling,
        'Savgol filter': savgol,
        'Gaussian smoothing': gaussian}

    num_cols = ['power', 'temperature', 'wind_speed', 'wind_direction']
    transform_plot, method_picker,  controllers = \
    vpw.plot_transform_time_series(df.loc['2014-01': '2014-02', num_cols],
                                   methods,
                                   columns={'description': 'Attribute'},
                                   kwargs_method={'description': 'Method'},
                                   controllers=[{'widget': 'IntSlider',
                                                 'max': 30, 'min': 1,
                                                 'value':10,
                                                 'description': '# Time units',},
                                                {'widget': 'RadioButtons',
                                                 'options': [('Days', 'D'), ('Hours', 'h')],
                                                 'value': 'h',
                                                 'description': 'Time unit'}])

    # organize controllers
    ui = widgets.VBox([widgets.HBox([method_picker, controllers[0]]), 
                    widgets.HBox([controllers[1], controllers[2]])])

    i = widgets.interactive_output(transform_plot, dict(method=method_picker, column=controllers[0], win=controllers[1], time_unit=controllers[2]))

    return ui, i


def plot_stl(df):
    """Render interactive panel to test seasonal trend decomposition with different settings

    Args:
        df: dataframe. Main dataset

    Returns:

    """
    periods = {'D': 365, 'H': 24, 'W': 7}
    #-------- DEFINE WIDGETS
    choice = widgets.RadioButtons(options=[('Seasonal', 'D'), ('Daily', 'H')],
                                  value='D',
                                  description='Decomposition period',
                                  style={'description_width': 'initial'})

    # define slider for date range
    selection_range_slider = define_slider_date_range(df.index.min(),
                                                      df.index.max())
    rnd_start = (selection_range_slider
                 .options[np.random.choice(len(selection_range_slider.options) - 366 * 3)][1])
    selection_range_slider.value = (rnd_start, rnd_start + np.timedelta64(3 * 365, 'D'))

    columns=['temperature', 'power']
    col_selector = widgets.Dropdown(options=[(vis_tools.prettify(c), c) for c in columns],
                           value='temperature',
                           description='Attribute')

    # DEFINE PLOTTING FUNCTION
    def stl_decompose(col, c, d):
        df_resampled = df.resample(c)[col].median().interpolate().loc[d[0]: d[1]]

        df_stl = stl_decomposition(df_resampled, periods[c])
        
        fig = vp.plot_multiple_time_series([df_stl[c] for c in df_stl.columns],
                                     series_names=df_stl.columns,
                                     trace_specs={'widths': [1, 5, 1, 1]},
                                     same_plot=False,
                                     show_legend=False,
                                     kwargs_subplots={'rows': 4, 'cols': 1})
        fig.update_layout(height=500)
        fig.show()

    return stl_decompose, choice, selection_range_slider, col_selector


def plot_boxplot_outliers_demo(df):
    """Plot schematics illustrating boxplot outliers

    Args:
        df: series. Pandas series with attribute to use

    Returns:

    """
    df = df[df.temperature > -15].temperature.copy()
    df.iloc[np.random.choice(len(df), 10)] = np.random.choice(np.arange(-12, -10), 10)

    plt.figure(figsize=(12, 4))
    plt.boxplot(df, vert=False, notch=True)
    ptiles = np.percentile(df, [25, 75])
    iqr_color = sb.xkcd_rgb['cool blue']
    plt.plot(ptiles, np.repeat(1.25, 2), color=iqr_color, linestyle='--')

    for k, v in enumerate(['<', '>']):
        plt.scatter(ptiles[k], 1.25, marker=v, color=iqr_color)
        
    plt.annotate('IQR', (np.mean(ptiles), 1.25), fontsize=18, color=iqr_color,
                 horizontalalignment='center', verticalalignment='bottom')

    ptile_cols = sb.xkcd_rgb['grass green']
    for k, v in enumerate(['25', '75']):
        plt.plot(np.repeat(ptiles[k], 2), [0.5, 1.1], color=ptile_cols, linestyle='--')
        plt.annotate(f'{v}th percentile', (ptiles[k]*1.01, 0.5), fontsize=18,
                     color=ptile_cols, horizontalalignment='right' if k == 0 else 'left')

    outliers = boxplot_outliers(df, 1.5)[1]
    outlier_cols = sb.xkcd_rgb['blood red']
    for k, v in enumerate([('lower', '25', '-'), ('upper', '75', '+')]):
        plt.plot(np.repeat(outliers[k], 2), [1, 1.4], color=outlier_cols, linestyle='--')
        plt.annotate(f'{v[0]} fence\n{v[1]}th ptile {v[2]} 1.5 * IQR', 
                     (outliers[k]*1.01, 1.25), color=outlier_cols, fontsize=18, 
                     horizontalalignment='right' if k == 0 else 'left')
    plt.axis('off')


def get_outlier_events(df):
    """Define interactive object to get outliers based on boxplot outlier definition settings. The signal is first
    detrended using STL by resampling it to a daily frequency, doing the STL decomposition and then upsampling to 10
    minute resolution. This procedure is used to speed up the process.

    Args:
        df: dataframe. Dataset

    Returns:

    """
    # resample to daily base
    df0 = df.resample('D').temperature.agg(np.mean).reset_index().set_index('datetime')

    # perform stl decomposition
    df_stl = stl_decomposition(df0.temperature.dropna(), 365)

    # upsample to 10min frequency and add to input dataset
    df['stl_temperature'] = (df_stl
                            .resample('10min')
                            .seasonal
                            .mean()
                            .interpolate())
    df['residual'] = df.temperature - df.stl_temperature

    n_iqr = widgets.FloatSlider(value=1.5, min=1, max=5, step=0.5,
                                description='Number of IQRs (alpha)',
                                style={'description_width': 'initial'},
                                continuous_update=False)

    int_mins = widgets.IntText(value=20, min=10, max=120, step=10,
                                description='Time interval (in minutes) for event merging',
                                style={'description_width': 'initial'},
                                continuous_update=False,
                                layout=widgets.Layout(width='50%'))

    return df, n_iqr, int_mins


def plot_outlier_events(df, outlier_events):
    """Render selected outlier event from table

    Args:
        df: dataframe. input dataset
        outlier_events: dataframe. Dataset with details on outlier events

    Returns:

    """
    # DEFINE WIDGETS
    e_id = widgets.Dropdown(options=outlier_events['Event ID'].unique(),
                            description='Event ID',
                            value=0)
    flank = widgets.IntSlider(value=2, min=1, max=30, step=1,
                              description='Flank duration (days)',
                              continuous_update=False)

    # DEFINE PLOTTING FUNCTION
    def visualize_outlier_event(e, f):
        evt = outlier_events[outlier_events['Event ID'] == e].iloc[0]
        df_sub = df.loc[evt.Start: evt.End]
        df_sub_flanks = df.loc[pd.to_datetime(evt.Start) - np.timedelta64(f, 'D'): 
                        pd.to_datetime(evt.End) + np.timedelta64(f, 'D')]
        cols = sb.color_palette('Set1')[:2]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Timeseries', 'Distribution'])

        # main plot
        fig0 = vp.plot_multiple_time_series([d.temperature for d in [df_sub_flanks, df_sub]],
                                            show_legend=False, trace_specs={'width': [1, 3], 'color': cols})

        # highlight outlier points of event
        fig0b = vp.plot_multiple_scatter(df_sub_flanks[df_sub_flanks.label_outlier > 0].reset_index(), 
                                         x='datetime', y='temperature',
                                         show_legend=False, trace_specs={'color': cols[0]})
        # highlight outlier points not part of event
        fig0c = vp.plot_multiple_scatter(df_sub[df_sub.label_outlier > 0].reset_index(), 
                                         x='datetime', y='temperature', 
                                     show_legend=False,
                                     trace_specs={'color': cols[1]})
        [fig.add_trace(d, row=1, col=1) for d in fig0.data]
        fig.add_trace(fig0b.data[0], row=1, col=1)
        fig.add_trace(fig0c.data[0], row=1, col=1)

        # make plot with boxplot distribution of values
        fig1 = vp.boxplot(df.residual,
                          trace_specs={'color': cols[0]},
                          box_specs={'boxpoints': 'outliers'})
        df_sub['x'] = 'residual'
        # highlight outlier poitns in event
        fig1b = vp.plot_multiple_scatter(df_sub, 'x', 'residual',
                                        trace_specs={'color': cols[1]})
        fig.add_trace(fig1.data[0], row=1, col=2)
        fig.add_trace(fig1b.data[0], row=1, col=2)
        fig.update_layout(showlegend=False)
        fig.show()

    return visualize_outlier_event, e_id, flank
    

def plot_normalization(df, N_IQR):
    """Render interactive panel to test effect of normalization type

    Args:
        df: dataframe. Input dataset
        N_IQR: int. Number of IQR to use for boxplot outlier definition

    Returns:

    """
    # function to exclude outliers
    def exclude_outliers(x):
        return x[~np.array(boxplot_outliers(x.copy(), N_IQR)[0])]

    # define normalization methods
    def minmax(x, c, remove_outliers):
        x = x[c]
        if remove_outliers:
            x = exclude_outliers(x)
        return (x - x.min()) / (x.max() - x.min())
    def zscore(x, c, remove_outliers):
        x = x[c]
        if remove_outliers:
            x = exclude_outliers(x)
        return (x - x.mean()) / x.std()
        
    methods = {
        'MinMax': minmax,
        'z-Score': zscore}

    transform_plot, method_picker, controllers = \
        vpw.plot_transform_time_series(df.resample('H')[['temperature', 'power', 'wind_speed']].mean(),
                                       methods,
                                       columns={'options': [(vis_tools.prettify(c), c)
                                                            for c in ['temperature', 'power', 'wind_speed']],
                                                'widget': 'Dropdown',
                                                'description': 'Attribute'},
                                       controllers=[{'widget': 'RadioButtons',
                                                     'options': [True, False],
                                                     'value':False,
                                                     'description': 'Remove outliers'}],
                                       plot_original=False,
                                       kwargs_ts={'trace_specs': {'cmap': 'Dark2', 'alpha': 0.75, 'width': 1}})

    return transform_plot, method_picker, controllers


def plot_fleet_outliers(df):
    """Render interactive panel showing outliers detected using fleet method

    Args:
        df: dataframe. Input dataset

    Returns:

    """
    # define order in which events show up in dropdown menu
    def set_event_order(evt_id):
        return np.append(np.append(evt_id, 'random'),
                         np.setdiff1d(fleet_outliers['Event ID'],
                                      evt_id))
    # pick random event
    def get_rnd_event():
        return fleet_outliers.sample(1).iloc[0]

    fleet_outliers = get_fleet_outliers(df)

    slider = widgets.IntSlider(value=6, min=1, max=24 * 7, step=1,
                               description='Flanks (hours)', 
                               continuous_update=False,
                               style={'description_width': 'initial'})
    evt_id = fleet_outliers['Event ID'].values[0]
    events = widgets.Dropdown(options=set_event_order(evt_id),
                              value=evt_id,
                              description='Outlier\nevent')

    # change order of outlier event
    def events_on_value_change(change):
        if change['new'] == 'random':
            rnd_evt = get_rnd_event()
            evt_id = rnd_evt['Event ID']
        else:
            evt_id = change['new']
        events.options = set_event_order(evt_id)
        events.value = evt_id
        slider.value = 6
    events.observe(events_on_value_change, names='value')

    # DEFINE PLOTTING FUNCTION
    def plot_outlier(evt, flank):
        if evt == 'random':
            rnd_evt = get_rnd_event()
        else:
            rnd_evt = fleet_outliers[fleet_outliers['Event ID'] == evt].iloc[0]

        ex = df.loc[rnd_evt.Start - np.timedelta64(flank, 'h'): 
                    rnd_evt.End + np.timedelta64(flank, 'h')]

        turbine_order = np.append(rnd_evt.turbine, 
                                  np.setdiff1d(df.turbine.unique(), 
                                               rnd_evt.turbine))
        fig = vp.plot_multiple_time_series([ex.loc[ex.turbine == t, 'power']
                                            for t in turbine_order],
                                           series_names=turbine_order,
                                           trace_specs={'width': [3, 1, 1, 1]},
                                           show_time_slider=False)
        vpt.add_shape(fig, 
                      x0=rnd_evt.Start, x1=rnd_evt.End, 
                      y0=ex.power.min(), y1=ex.power.max(),
                      sh_type='rect', sh_specs={'alpha': 0.2})
        fig.show()

    return plot_outlier, events, slider


def plot_imputation(df, missing_events):
    """Render interactive panel to showcase linear and fleet-based imputation

    Args:
        df: dataframe. input dataset
        missing_events: dataframe. summary of missing data events

    Returns:

    """

    # define order in which events show up in dropdown menu
    def set_event_order(evt_id):
        return np.append(np.append(evt_id, 'random'),
                         np.setdiff1d(missing_events.event_id,
                                      evt_id))
    # pick random event with given duration
    def get_rnd_event(s):
        return (missing_events[missing_events['Duration (min)'] == s].sample(1).iloc[0])

    slider = widgets.IntSlider(value=30, min=10, max=missing_events['Duration (min)'].max(), step=10,
                               description='Event duration (minutes)', 
                               continuous_update=False,
                               style={'description_width': 'initial'})
    evt_id = missing_events.event_id.values[0]
    events = widgets.Dropdown(options=set_event_order(evt_id),
                              value=evt_id,
                              description='Missing data event')

    turbine_selector = widgets.ToggleButtons(
        options=np.arange(1, 4),
        value=3,
        description='# turbines')

    # change order of events in dropdown menu
    def events_on_value_change(change):
        if change['new'] == 'random':
            evt_id = get_rnd_event(slider.value).event_id
        else:
            evt_id = change['new']
        events.options = set_event_order(evt_id)
        events.value = evt_id
    events.observe(events_on_value_change, names='value')

    # change event on change of duration
    def slider_on_value_change(change):
        durations = missing_events['Duration (min)'].unique()
        nearest = durations[np.argmin(np.abs(durations - change['new']))]
        slider.value = nearest
        evt_id = get_rnd_event(nearest).event_id
        events.options = set_event_order(evt_id)
        events.value = evt_id
    slider.observe(slider_on_value_change, names='value')

    # DEFINE PLOTTING FUNCTION
    def imputate_missing(evt, nt, s):
        # if evt == 'random':
        #     rnd_evt = get_rnd_event(s)
        # else:
        rnd_evt = missing_events[missing_events.event_id == evt].iloc[0]

        # define start and end times using a 2 hours flanking window
        win = 120
        st = rnd_evt.Start - np.timedelta64(win, 'm')
        en = rnd_evt.End + np.timedelta64(win, 'm')
        df0 = df.loc[st: en].copy()

        # make linear interpolation
        df0['linear_interpolation'] = df0.groupby('turbine').wind_speed.transform(lambda x: x.interpolate())

        # define order of turbines
        turbines = np.append(rnd_evt.turbine,
                             np.random.choice(np.setdiff1d(missing_events.turbine.unique(), 
                                                           rnd_evt.turbine), nt, replace=False))

        # make fleet-based inteprolation
        df_fleet = (df[df.turbine.isin(turbines)].loc[df[df.missing_event == rnd_evt.missing_event].index]
                    .groupby('datetime')
                    .wind_speed
                    .mean()
                    .reset_index(name='fleet_mean'))

        df0 = df0.join(df_fleet.set_index('datetime'))
        df0['fleet_mean'] = df0.wind_speed.combine_first(df0.fleet_mean)

        # make plot
        colors = sb.color_palette('Set1')[:3]
        colors = [vpt.to_rgba(c) for c in colors]
        colors = colors[:1] + [vpt.to_rgba([0, 0, 0], 0.3)] * nt + colors[1: ] + colors[:1]
        widths = [2] + [1] * nt + [3] * 3
        linestyles = ['solid'] + ['solid'] * (nt) + ['solid', 'solid', 'solid']

        id_evt = ((df0.turbine == rnd_evt.turbine)
                  & (df0.index >= (rnd_evt.Start - np.timedelta64(10, 'm')))
                  & (df0.index <= (rnd_evt.End + np.timedelta64(10, 'm'))))

        fig = vp.plot_multiple_time_series(
            ([df0[df0.turbine == t].wind_speed for t in turbines]
             + [df0.loc[id_evt, 'linear_interpolation']]
             + [df0.loc[id_evt, 'fleet_mean']]
             + [df0.loc[id_evt, 'wind_speed_original']]),
            series_names=np.append(turbines, ['Linear interpolation', 'Fleet mean', 'Original trace']),
            trace_specs={'width': widths, 'color': colors, 'dash': linestyles},
            show_time_slider=False)
        for x0 in [rnd_evt.Start, rnd_evt.End]:
            vpt.add_shape(fig, x0=x0, 
                          y0=df0.wind_speed.min(), y1=df0.wind_speed.max(),
                          sh_specs={'alpha': 0.5}) 
        fig.show()

    return imputate_missing, events, slider, turbine_selector


def plot_pattern_imputation(df):
    """Render interactive panel to test different settings on pattern imputation

    Args:
        df: dataframe. input dataset

    Returns:

    """
    features = widgets.Dropdown(options=[(vis_tools.prettify(c), c)
                                         for c in ['wind_speed', 'temperature']],
                                value='temperature',
                                description='Feature')
    slider = widgets.IntSlider(value=2, min=1, max=48, step=1,
                               description='Event duration (hours)', 
                               continuous_update=False,
                               style={'description_width': 'initial'})

    slider_evt_st = widgets.IntSlider(value=0, min=0, max=48, step=1,
                               description='Shift event start (hours)', 
                               continuous_update=False,
                               style={'description_width': 'initial'})

    # define missign data event and subset
    evt = df[df.index == '2016-08-25 07:30:00'].sample(1).iloc[0]

    evt_start = pd.to_datetime('2016-08-25 07:30:00')
    df_arima = (df.loc[evt_start - np.timedelta64(365, 'D'): evt_start + np.timedelta64(2, 'D')])
    df_hour = df_arima[['temperature', 'wind_speed']].resample('H').mean()
    df0 = df_hour[(df_hour.index >= evt.name - np.timedelta64(3, 'D')) &
                  (df_hour.index <= evt.name + np.timedelta64(2, 'D'))].copy()
    
    def _plot_pattern_imputation(v, d, s):
        df0_sub = df0.copy()
        df0_sub['original'] = df0_sub[v]

        # define adjusted start of event based on shifting amount
        evt_start_shifted = evt_start - np.timedelta64(s, 'h')
        df_id = (df0_sub.index >= evt_start_shifted) & \
            (df0_sub.index < evt_start_shifted + np.timedelta64(d, 'h'))
        df0_sub.loc[df_id, v] = np.nan

        # do ARIMA-based pattern imputation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = sm.tsa.statespace.SARIMAX(df_hour.loc[evt_start_shifted - np.timedelta64(15, 'D'):evt_start_shifted, v],
                                            order=(0, 1, 0),
                                            seasonal_order=(1, 1, 0, 24),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit(disp=0)
            data_imputed = results.get_forecast(steps=int(df0_sub[v].isnull().sum())).predicted_mean.values

        # add imputed values to proper rows in dataset
        df0_sub['imputed'] = np.nan
        df_id2 = (df0_sub.index >= (evt_start_shifted - np.timedelta64(1, 'h'))) & \
            (df0_sub.index < (evt_start_shifted + np.timedelta64(d, 'h') + np.timedelta64(1, 'h')))
        df0_sub.loc[df_id2, 'imputed'] = df0_sub.loc[df_id2, 'original']
        df0_sub.loc[df0_sub[v].isnull(), 'imputed'] = data_imputed

        df0_sub['original_plot'] = np.nan
        df0_sub.loc[df_id2, 'original_plot'] = df0_sub.loc[df_id2, 'original']

        fig = vp.plot_multiple_time_series([df0_sub.original, df0_sub.imputed],
                                           series_names=['Original', 'Imputed'],
                                           show_time_slider=False,
                                           trace_specs={'width': [1, 2]})

        st, en = df0_sub.loc[df_id2].index.min(), df0_sub.loc[df_id2].index.max()
        for x0 in [st, en]:
            vpt.add_shape(fig, x0=x0, 
                          y0=df0_sub[v].min(), y1=df0_sub[v].max(),
                          sh_specs={'alpha': 0.5}) 
        fig.show()

    return _plot_pattern_imputation, features, slider, slider_evt_st

