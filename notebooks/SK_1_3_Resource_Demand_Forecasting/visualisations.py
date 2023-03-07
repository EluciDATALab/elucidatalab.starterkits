# ©, 2022, Sirris
# owner: HCAB

import pandas as pd
import numpy as np
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from starterkits.visualization import vis_plotly as vp
#from elucidata.tools.visualization import vis_plotly as vp
from starterkits.visualization import vis_plotly_widgets_tools as vpwt

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime,timedelta
import multiprocessing
from joblib import Parallel, delayed

from support import * # get_months


def plot_real_estimated_power(df_real_values,array_predicted_values, ax=None):
    """
    plot the real and estimated values of the global active power. The values are resampled on daily basis

    df_real_values: dataframe containing the real values
    array_predicted_values: array containing the predicted values
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    df_copy = df_real_values.copy()
    df_copy['Predicted'] = array_predicted_values
    df_copy = df_copy.resample('D').median().copy()
    df_copy.loc[:,'consumption'].plot(figsize=(16,5), ax=ax)
    df_copy.loc[:,'Predicted'].plot(figsize=(16,5), legend=True, ax=ax)
    ax.legend(['Observed', 'Predicted'])
    ax.set_title('Evolution global active power')
    ax.set_ylabel('Active Power [Wh]');
    return ax


def plot_MAE(scores_baseline,scores_rfr,scores_svr,title):
    """
    plot the MAE of each model.

    scores_baseline: MAE of the dummy model
    scores_rfr: MAE of the random forest model
    scores_svr: MAE of the support vector regression model
    """

    plt.figure(figsize=(14,5))
    plt.plot(scores_baseline)
    plt.plot(scores_rfr)
    plt.plot(scores_svr)
    plt.xlabel('Month')
    plt.ylabel('MAE')
    plt.legend(['Dummy', 'Random Forest Regression', 'Support Vector Regression'], loc='upper right')
    plt.title(title)

def plot_month_with_weekends_and_holidays_highlighted(df, start, end):
    fr_holidays = holidays.France() #Change variable name in fr_holidays
    df['Weekday'] = df.index.dayofweek

    fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', figsize=(15,5))

    df[start:end]['Global_active_power'].resample('4H').sum().plot(ax=ax,title='Evolution global active power')

    for date in df[start:end].index:
        if df.loc[date]['Weekday'] == 5 or df.loc[date]['Weekday'] == 6:
            plt.axvspan(date, date + timedelta(hours=1), color='orange', alpha=0.5, lw=None)
        if date in fr_holidays:
            plt.axvspan(date, date + timedelta(hours=1), color='red', alpha=0.5, lw=None)
    ax.set_ylabel('consumption energy [Wh]');

def plot_year_with_temperature(df, start, end):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharey='row', figsize=(15,5))

    plot_df = df[start:end].resample('D').agg({'Global_active_power':'sum','temperature':'mean'})

    ax1.set_ylabel('Active Power [Wh]');
    plot_df['Global_active_power'].plot(ax=ax1, color='tab:blue')

    ax2 = ax1.twinx()

    ax2.set_ylabel('temperature')
    plot_df['temperature'].plot(ax=ax2, color='tab:orange')

    ax1.set_title('Evolution global active power & temperature')

    fig.tight_layout()
    plt.show()

def plot_day_with_submeters(df, start, end):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharey='row', figsize=(15,10))

    df[start:end]['Global_active_power'].plot(title='Evolution global active power', ax=ax[0])
    ax[0].set_ylabel('consumption energy [kWmin]');

    df[start:end][['Sub_metering_1','Sub_metering_2','Sub_metering_3']].plot(title='Evolution energy consumption per sub metering', ax=ax[1]);
    ax[1].set_ylabel('consumption energy [Wh]');

def plot_weekly_data(df):
    start = '2009-12-01 00:00:00'
    end = '2009-12-07 23:23:59'


    plot_df = df[start:end]['Global_active_power']
    fig = vp.plot_multiple_time_series([plot_df],
                                       show_time_slider=False, show_time_frame=False)
    fig.update_layout(yaxis_title='Global active power [Wh]',
              xaxis_title='Date',
              title='Evolution global active power',
              font={'size': 18},
              showlegend=True)
    fig.show()


def plot_correlation(df):

    def make_plot(resample_rate):

        if resample_rate is None:
            resampled_df = df.copy()
        else:
            resampled_df = df.resample(resample_rate).agg({'Global_active_power':'sum','temperature':'mean'})
        corr = resampled_df.corr()
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig = px.imshow(corr, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        fig.show()

    controller= vpwt.get_controller({'widget': 'Dropdown',
                                     'options': [('No resampling', None),
                                                 ('1 Hour', 'H'),
                                                 ('8 Hours', '8H'),
                                                 ('1 Day', 'D'),
                                                 ('1 Week', '7D'),
                                                 ('30 days', '30D')],
                                     'value': 'H', 'description': 'Resample rate',
                                     'style': {'description_width': 'initial'}})

    interact(make_plot, resample_rate=controller)


def plot_effects_resampling(data, show_time_frame=False):
    def rolling(x, c, window):
        # Use a rolling window
        x = x[c]
        return x.rolling(f'{window}T').mean()

    methods = {'Rolling': rolling}

    sampled_data = data.loc['2008-04-01 00:00:00': '2008-04-30 23:59:59'][
        ['Global_active_power']]  # .sample(100000).sort_index()

    transform_plot, method_picker, controllers = \
        vpw.plot_transform_time_series(sampled_data, methods,
                                       controllers=[{'widget': 'Dropdown',
                                                     'options': [('30 minutes', 30), ('1 Hour', 60),
                                                                 ('4 Hours', 240)],
                                                     'value': 30,
                                                     'description': 'Window size for resampling',
                                                     'style': {'description_width': 'initial'}}],
                                       kwargs_ts={'show_time_frame': show_time_frame, 'show_time_slider': False})
    interact(transform_plot, method=fixed('Rolling'), column=fixed('Global_active_power'), win=controllers[1]);


def plot_temperature_power_one_year_old(new_data):
    plot_df = new_data[['Global_active_power', 'temperature']].resample('D').agg({'Global_active_power': 'sum',
                                                                                  'temperature': 'mean'})
    plot_df = plot_df.dropna()
    scaler = MinMaxScaler()
    plot_df[['Global_active_power', 'temperature']] = scaler.fit_transform(
        plot_df[['Global_active_power', 'temperature']])
    fig = vp.plot_multiple_time_series([plot_df['Global_active_power'], plot_df['temperature']],
                                       show_time_slider=False, show_time_frame=False)  # , same_plot=False)
    fig.show()


def plot_temperature_power_one_year(new_data, show_time_frame=False):
    def normalize_data(x, c, normalize):
        print(c, normalize)
        x = x.copy()
        if normalize:
            scaler = MinMaxScaler()
            x[c] = scaler.fit_transform(np.array(x[c]).reshape(-1, 1))

        return x

    plot_df = new_data[['Global_active_power', 'temperature']].resample('D').agg({'Global_active_power': 'mean',
                                                                                  'temperature': 'mean'})
    plot_df = plot_df.dropna()

    both_columns = ['Global_active_power', 'temperature']

    controller_norm = vpwt.get_controller({'widget': 'Checkbox', 'value': False, 'description': 'Normalize data?'})
    controller_columns = vpwt.get_controller({'widget': 'RadioButtons',
                                              'options': [('Plot active power & temperature', both_columns),
                                                          ('Only plot active power', ['Global_active_power']),
                                                          ('Only plot temperature', ['temperature'])],
                                              'value': both_columns, 'description': ' '})

    def make_plot(normalize):
        df = plot_df[both_columns].copy()
        if normalize:
            scaler = MinMaxScaler()
            df[both_columns] = scaler.fit_transform(df)
        series_to_plot = [df[col] for col in both_columns]
        fig = vp.plot_multiple_time_series(series_to_plot,
                                           show_time_slider=False, show_time_frame=False)  # , same_plot=False)
        fig.show()

    # sampled_data = data.loc['2008-04-01 00:00:00': '2008-04-30 23:59:59'][['Global_active_power']]#.sample(100000).sort_index()

    interact(make_plot, normalize=controller_norm);


def plot_consumption_with_holidays_weekends_old2(new_data):
    resample_rate_hour = 4
    df = new_data[['Global_active_power']].resample(f'{resample_rate_hour}H').agg({'Global_active_power': 'sum'})
    years = np.arange(2007, 2010)

    def make_plot(**controllers):
        # print(controllers)
        years = [int(descr[-4:]) for descr, val in controllers.items() if val]
        series_to_plot = []
        shapes = []

        colors = vpt.get_colors('Dark2', [0.5 for _ in years])

        for _i, year in enumerate(years):

            start_lastweek = f'{year}-12-24'
            end_lastweek = f'{year}-12-31'
            df_lastweek = df.loc[start_lastweek:end_lastweek]
            last_sunday = None
            _j = 0
            #
            while last_sunday is None:
                date = df_lastweek.index[_j]
                if date.dayofweek == 6:
                    last_sunday = date

                _j += 1
            start_day = last_sunday - pd.Timedelta('27D')
            christmas_idx = (24 - (last_sunday.day - 28))
            # print(last_sunday, last_sunday.day, christmas_idx)

            plot_df = df[start_day:last_sunday + pd.Timedelta('23H') + pd.Timedelta('59m')].reset_index(drop=True)
            plot_df['index'] = np.arange(0, 28, resample_rate_hour / 24)
            plot_df = plot_df.set_index('index')

            d = dict(type="rect",
                     xref="x",
                     # y-reference is assigned to the plot paper [0,1]
                     yref="paper",
                     x0=christmas_idx,
                     y0=0,
                     x1=christmas_idx + 1,  # + timedelta(hours=4),
                     y1=1,
                     fillcolor=colors[_i],
                     opacity=0.25,
                     layer="below",
                     line_width=1)
            shapes.append(d)
            # print(plot_df)
            series_to_plot.append(plot_df['Global_active_power'])

        series, _, trace_specs, trace_cols = \
            vpt.define_figure_specs(series_to_plot, None, {},
                                    ['alpha', 'width', 'dash', 'color'])

        trace_specs['scatter_kwargs'] = {}
        plot_assistant = {'series': series, 'series_names': years,
                          'trace_cols': trace_cols, 'trace_specs': trace_specs,
                          'mode': ['lines'], 'scatter_kwargs': {}}
        # print(trace_cols, trace_specs)

        n_series = len(series)
        fig = go.Figure([vp._make_timeseries(k, plot_assistant) for k in np.arange(n_series)])

        fig.update_layout(yaxis_title='Global active power [Wh]',
                          xaxis_title='Day number',
                          title='Last 4 weeks of the year (with days of the week aligned)',
                          font={'size': 18},  # 'color': "#7f7f7f"},
                          showlegend=True)
        # fig = vp.plot_multiple_time_series(series_to_plot, show_time_slider=False,
        #                                   show_time_frame=False)

        # shapes = []
        # fr_holidays = holidays.France()
        for idx in [5, 12, 19, 26]:  # np.arange(0, 28, 1/6):
            # day_of_week = int(np.floor(idx)) % 7
            # if day_of_week in (5, 6):
            # ax.axvspan(date, date + timedelta(hours=1), color='orange', alpha=0.5, lw=None)
            d = dict(type="rect",
                     xref="x",
                     # y-reference is assigned to the plot paper [0,1]
                     yref="paper",
                     x0=idx,
                     y0=0,
                     x1=idx + 2,  # + timedelta(hours=4),
                     y1=1,
                     fillcolor="yellow",
                     opacity=0.5,
                     layer="below",
                     line_width=1)
            shapes.append(d)

        fig.update_layout(
            shapes=shapes
        )

        fig['layout']['xaxis'].update(range=[0, 28])
        fig['layout']['yaxis'].update(range=[0, 20000])

        fig.show()

    checked = {2007: True, 2008: False, 2009: False}
    controllers = {f'plot_{year}': vpwt.get_controller({'widget': 'Checkbox', 'value': checked[year],
                                                        'description': f'{year}'}) for year in years}

    interact(make_plot, **controllers)


def plot_consumption_with_holidays_weekends(new_data):
    resample_rate_hour = 4
    df = new_data[['Global_active_power']].resample(f'{resample_rate_hour}H').agg({'Global_active_power': 'sum'})
    years = np.arange(2007, 2010)

    def make_plot(year):
        # print(controllers)
        series_to_plot = []
        shapes = []

        colors = vpt.get_colors('Dark2', [0.5 for _ in years])

        start_dec = f'{year}-12-01 00:00:00'
        end_dec = f'{year}-12-31 23:59:59'

        plot_df = df.loc[start_dec:end_dec]  # .reset_index(drop=True)
        # plot_df['index'] = np.arange(0, 28, resample_rate_hour / 24)
        # plot_df = plot_df.set_index('index')

        fr_holidays = holidays.France()

        for day in plot_df.index:
            if day.dayofweek == 5 and day.hour == 0:
                d = dict(type="rect",
                         xref="x",
                         # y-reference is assigned to the plot paper [0,1]
                         yref="paper",
                         x0=day,
                         y0=0,
                         x1=day + timedelta(hours=48),
                         y1=1,
                         fillcolor='yellow',
                         opacity=0.25,
                         layer="below",
                         line_width=1)
                shapes.append(d)
            if day.date() in fr_holidays and day.hour == 0:
                d = dict(type="rect",
                         xref="x",
                         # y-reference is assigned to the plot paper [0,1]
                         yref="paper",
                         x0=day,
                         y0=0,
                         x1=day + timedelta(hours=24),
                         y1=1,
                         fillcolor='orange',
                         opacity=0.5,
                         layer="below",
                         line_width=1)
                shapes.append(d)
        # print(plot_df)
        series_to_plot.append(plot_df['Global_active_power'])

        series, _, trace_specs, trace_cols = \
            vpt.define_figure_specs(series_to_plot, None, {},
                                    ['alpha', 'width', 'dash', 'color'])

        trace_specs['scatter_kwargs'] = {}

        plot_assistant = {'series': series, 'series_names': [year],
                          'trace_cols': trace_cols,
                          'mode': ['lines'], 'scatter_kwargs': [{}], 'trace_specs': trace_specs}
        # print(trace_cols, trace_specs)

        n_series = len(series)
        fig = go.Figure([vp._make_timeseries(k, plot_assistant) for k in np.arange(n_series)])

        fig.update_layout(yaxis_title='Global active power [Wh]',
                          xaxis_title='Day number',
                          title=f'Power consumption of December {year}',
                          font={'size': 18},  # 'color': "#7f7f7f"},
                          showlegend=True)
        # fig = vp.plot_multiple_time_series(series_to_plot, show_time_slider=False,
        #                                   show_time_frame=False)

        # shapes = []
        '''
        for idx in [5, 12, 19, 26]:#np.arange(0, 28, 1/6):
            #day_of_week = int(np.floor(idx)) % 7
            #if day_of_week in (5, 6):
                #ax.axvspan(date, date + timedelta(hours=1), color='orange', alpha=0.5, lw=None)
                d = dict(type="rect",
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=idx,
                    y0=0,
                    x1=idx+2,# + timedelta(hours=4),
                    y1=1,
                    fillcolor="orange",
                    opacity=0.5,
                    layer="below",
                    line_width=1)
                shapes.append(d)
        '''

        fig.update_layout(
            shapes=shapes
        )

        # fig['layout']['xaxis'].update(range=[0, 28])
        fig['layout']['yaxis'].update(range=[0, 20000])

        fig.show()

    checked = {2007: True, 2008: False, 2009: False}
    controller_year = vpwt.get_controller({'widget': 'RadioButtons', 'options': [2007, 2008, 2009],
                                           'value': 2007, 'description': 'Year'})

    interact(make_plot, year=controller_year)


def plot_yearly_data(new_data):
    resampling_rate_days = 1

    years = np.arange(2007, 2010)

    def make_plot(resampling_rate):  # , **controllers):
        resampling_rate_days = int(resampling_rate[:-1])
        df = new_data[['Global_active_power']].copy()
        # years = [int(descr[-4:]) for descr, val in controllers.items() if val]
        series_to_plot = []

        def is_leap_year(year):
            return (year % 4 == 0 and not year % 100 == 0) or year % 1000 == 0

        colors = vpt.get_colors('Dark2', [0.5 for _ in years])

        for _i, year in enumerate(years):

            start_year = f'{year}-01-01 00:00:00'
            end_year = f'{year}-12-31 23:59:59'
            plot_df = df.loc[start_year:end_year].resample(resampling_rate).agg({'Global_active_power': 'sum'})

            number_of_days = 365
            if is_leap_year(year):
                number_of_days += 1

            days = np.arange(0, number_of_days, resampling_rate_days)
            month = days / 30 + 1
            plot_df['month'] = month
            plot_df = plot_df.set_index('month')
            # print(plot_df)
            series_to_plot.append(plot_df['Global_active_power'])

        series, _, trace_specs, trace_cols = \
            vpt.define_figure_specs(series_to_plot, None, {},
                                    ['alpha', 'width', 'dash', 'color'])
        trace_specs['scatter_kwargs'] = {}

        _years = [str(year) for year in years]
        plot_assistant = {'series': series, 'series_names': _years,
                          'trace_cols': trace_cols, 'trace_specs': trace_specs,
                          'mode': ['lines'] * len(years)}

        n_series = len(series)
        fig = go.Figure([vp._make_timeseries(k, plot_assistant) for k in np.arange(n_series)])

        if resampling_rate_days > 1:
            suffix = 's'
        else:
            suffix = ''
        fig.update_layout(yaxis_title=f'Power consumed in {resampling_rate_days} day{suffix}  [Wh]',
                          xaxis_title='Month',
                          title=' ',
                          font={'size': 18},  # 'color': "#7f7f7f"},
                          showlegend=True)

        fig.show()

    # checked = {2007: True, 2008: False, 2009: False}
    # controllers_year_selection = {f'plot_{year}': vpwt.get_controller({'widget': 'Checkbox', 'value': checked[year],
    #                                                    'description': f'{year}',
    #                                                    'style': {'description_width': 'auto'}}) for year in years}

    controller_resampling = vpwt.get_controller({'widget': 'Dropdown',
                                                 'options': [('1 day', '1D'), ('3 days', '3D'), ('1 week', '7D')],
                                                 'value': '1D',
                                                 'description': 'Resampling rate',
                                                 'style': {'description_width': 'initial'}})
    interact(make_plot, resampling_rate=controller_resampling)  # , **controllers_year_selection)


def plot_auto_correlation(data):
    power = data['Global_active_power']

    steps = 30
    hours_a_day = 24
    corr = np.zeros(steps - 1)
    x = np.arange(1, steps)
    for i in x:
        corr[i - 1] = power.autocorr(i * hours_a_day)

    fig = px.line(x=x, y=corr)
    fig.update_layout(yaxis_title='Correlation',
                      xaxis_title='Days',
                      xaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
                      # title='Evolution global active power',
                      font={'size': 18},  # 'color': "#7f7f7f"},
                      showlegend=True)
    fig['layout']['yaxis'].update(range=[0.25, 0.5])
    fig.show()


METHODS = ['baseline', 'SVM', 'RandomForest']


def construct_layout(component_dict):
    controls = dbc.Card(children=[], body=True)
    for label,comp in component_dict.items():
        group = [dbc.FormGroup([comp
                               ])]
        controls.children.extend(group)
#     print(controls.children)
    return controls


def construct_param_layout(params_dict):
    params_card = dbc.Card(children=[], body=True)
    for params in params_dict:
        form = [dbc.FormGroup([
                dbc.Label(params['name']),
                dbc.Input(id=params['id'], type=params['type'], value=params['value']),
                dbc.FormText(params['text'], color='secondary'),])]
        params_card.children.extend(form)
    return params_card

def construct_parameter_layout(params_dict, method):

    if method == 'baseline':
        strategies = ['Train-test split']
        # description = {'previous day': 'Use the consumption of the day before as prediction'}
    else:
        strategies = ['1 month', '6 months', '1 year', '1 month year before', 'All months before', 'Train-test split']
    description = {'1 month': '1 month before',
                   '6 months': '6 months before',
                   '1 year': '1 year before',
                   'All months before': 'All months before',
                   '1 month year before': '1 month the year before',
                   'Train-test split': 'Train-test split'}
    # params_dict = dbc.Card(children=[], body=True)

    params_dict.extend(construct_param('Strategy', 'strategy', strategies[0], None,
                                       'Choose the training strategy',
                                       options=strategies, description=description))

    parameters_card = construct_dropdown_layout(params_dict)

    normalize_group = [dbc.FormGroup([dbc.Checkbox(id='normalize-checkbox',
                                                   key='Normalize', checked=True,
                                                   className="form-check-input"),
                                      dbc.Label("Normalize before training the model?",
                                                html_for="normalize-checkbox",
                                                className="form-check-label")
                                      ])]

    parameters_card.children.extend(normalize_group)
    return parameters_card

def construct_dropdown_layout(params_dict):
    params_card = dbc.Card(children=[], body=True)
    for params in params_dict:
        options = params['options']
        if 'description' in params.keys():
            description = params['description']
        else:
            description = {option: option for option in options}

        options=[{'label': description[option], 'value': option}  for _j, option in enumerate(options)]
        form = [dbc.FormGroup([
                dbc.Label(params['name']),
                dcc.Dropdown(id=params['id'], options=options, value=params['value']),
                dbc.FormText(params['text'], color='secondary'),])]
        params_card.children.extend(form)
    return params_card

def get_all_options(df):
    options = [{'label': 'All', 'value': 'All'}]
    for cols in df.columns:
        options.append({'label': cols, 'value': cols})
    return options

def construct_param(n, i, v, t, te, **kwargs):
    return [{'name': n, 'id': i, 'value': v, 'type': t, 'text': te, **kwargs}]


def get_regressor(method, **parameters_values):

    assert method in METHODS

    method_param_dict = get_method_param_layout(method)
    # print(parameters_values)
    #parameters_values = {n['id']:n['value'] for n in method_param_dict}
    if method == 'baseline':
        return 'baseline'
    elif method == 'SVM':

        #return SVR(C=10, gamma=0.01)
        return SVR(**parameters_values)
    else:
        if parameters_values['max_depth'] == 0:
            parameters_values['max_depth'] = None
        return RandomForestRegressor(random_state=42, **parameters_values)


def get_method_param_layout(method):
    if method == 'SVM':
        # options_kernel = ['rbf', 'linear', 'poly']
        options_C = [10, 100, 1000]
        options_gamma = [0.01, 0.1, 1]
        method_param_dict = construct_param('C', 'C', 1000, 'number',
                                       'Choose the C parameter', options=options_C)
        method_param_dict.extend(construct_param('Gamma', 'gamma', 0.01, "number",
                               'Choose the gamma parameter', options=options_gamma))
    elif method == 'RandomForest':
        options_n_estimators = [10, 25, 50, 100]
        options_max_depth = [0, 5, 10]
        method_param_dict = construct_param('Number of estimators', 'n_estimators', 100, 'number',
                                                     'Choose the number of estimators', options=options_n_estimators)
        method_param_dict.extend(construct_param('Max depth', 'max_depth', 0, "number",
                               'Choose the maximum depth of the tree (choose 0 to expand until all leaves are pure)',
                                                         options=options_max_depth))

    else:
        method_param_dict = []

    return method_param_dict


def get_method_name(method):
    assert method in METHODS
    if method == 'baseline':
        name = 'Baseline Model'
    elif method == 'SVM':
        name = 'Support Vector Regressor'
    elif method == 'RandomForest':
        name = 'Random Forest Regressor'
    return name

def get_output_layout(method, test_feats, start_date, end_date, target):
    y = pd.Series(test_feats[target], name='True value')
    fig = vp.plot_multiple_time_series([y], show_time_slider=True)
    #fig.update_xaxes(range=[start_date, end_date])
    fig['layout']['xaxis'].update(range=[start_date, end_date])
    fig.update_layout(yaxis_title="Consumption [Wh]")
    # add parameter layout
    comp_dict = {}

    comp_dict['Features plot'] = dcc.Graph(id='comparison_plot', figure=fig)

    return comp_dict

def get_method_parameters(params):
    parameters = {d['props']['children'][1]['props']['id']: d['props']['children'][1]['props']['value']
                  for d in params[0]['props']['children']
                  if 'id' in d['props']['children'][1]['props'].keys() and
                  d['props']['children'][1]['props']['id'] != 'strategy'}
    return parameters


def get_predictions(regr, strategy, new_feats, normalize):

    # multiprocessing does not work sometimes on some machines
    num_cores = 1  # multiprocessing.cpu_count()

    if strategy == '1 month':
        offset_test = 1
        window_size_test = 1
        window_size_train = 1

    if strategy == '6 months':
        offset_test = 1
        window_size_test = 1
        window_size_train = 6

    if strategy == '1 year':
        offset_test = 1
        window_size_test = 1
        window_size_train = 12

    if strategy == 'All months before':
        offset_test = 1
        window_size_test = 1
        window_size_train = 1

    if strategy == '1 month year before':
        offset_test = 11
        window_size_test = 1
        window_size_train = 1


    if strategy == 'Train-test split':
        offset = 0
    else:
        offset = offset_test + window_size_train
    offset_years = (offset + 11) // 12
    offset_months = offset % 12
    year = 2008 - offset_years
    month = (-offset_months) % 12 + 1
    start_date_test = pd.Timestamp(f'{year}-{month: 02d}-01 00:00:00')
    # start_date_test = pd.Timestamp(f'2008-01-01 00:00:00')

    # train_feats = new_feats[:start_date_test]
    train_feats = new_feats['2007-02-01':'2007-12-31'].copy()
    test_feats = new_feats[start_date_test:].copy()


    if regr == 'baseline':
        true_values = test_feats[target]
        pred = test_feats['Pastday']

    else:

        if normalize:
            scaler_features = MinMaxScaler()
            scaler_target = MinMaxScaler()
            train_feats[features] = scaler_features.fit_transform(train_feats[features])
            test_feats[features] = scaler_features.transform(test_feats[features])

            train_feats[[target]] = scaler_target.fit_transform(train_feats[[target]])
            test_feats[[target]] = scaler_target.transform(test_feats[[target]])


        months = get_months(test_feats)

        if strategy == 'All months before':
            max_window_size = len(months) - offset_test - 1 - window_size_test
            pred = Parallel(n_jobs=num_cores)(delayed(predictions_window)(regr, test_feats, 0, w, offset_test,
                                                                          window_size_test)
                                              for w in range(1, max_window_size))
            true_values = Parallel(n_jobs=num_cores)(delayed(get_true_values)(test_feats, 0, w,
                                                          offset_test, window_size_test)
                                                     for w in range(1, max_window_size))
            timestamps = Parallel(n_jobs=num_cores)(delayed(get_datetimes)(test_feats, 0, w,
                                                          offset_test, window_size_test)
                                                     for w in range(1, max_window_size))

            pred = np.concatenate(pred)
            true_values = np.concatenate(true_values)
            timestamps = np.concatenate(timestamps)

        elif strategy == 'Train-test split':

            regr.fit(train_feats[features], train_feats[target])
            pred = np.array(regr.predict(test_feats[features]))
            true_values = np.array(test_feats[target])
            timestamps = test_feats.index

        else:
            max_idx_month = len(months) - offset_test - 1 - window_size_test - window_size_train
            pred = Parallel(n_jobs=num_cores)(delayed(predictions_window)(regr, test_feats, m, window_size_train,
                                                                  offset_test, window_size_test)
                                              for m in range(0, max_idx_month))
            true_values = Parallel(n_jobs=num_cores)(delayed(get_true_values)(test_feats, m, window_size_train,
                                                                  offset_test, window_size_test)
                                                     for m in range(0, max_idx_month))

            timestamps = Parallel(n_jobs=num_cores)(delayed(get_datetimes)(test_feats, m, window_size_train,
                                                                  offset_test, window_size_test)
                                                     for m in range(0, max_idx_month))

            pred = np.concatenate(pred)
            true_values = np.concatenate(true_values)
            timestamps = np.concatenate(timestamps)

        if normalize:
            pred = scaler_target.inverse_transform(pred.reshape(-1, 1)).flatten()
            true_values = scaler_target.inverse_transform(true_values.reshape(-1, 1)).flatten()

        pred = pd.Series(pred, index=timestamps)
        true_values = pd.Series(true_values, index=timestamps)

    return pred, true_values


def run_app(new_feats, mode='external', port=8093, debug=True, height=1000, host='0.0.0.0'):
    global busy_computing, output_text, number_of_dots
    # app_params = {}
    # method_param_dict = {}
    initial_start_date = pd.Timestamp('2008-04-01 00:00:00')
    initial_end_date = pd.Timestamp('2008-04-30 00:00:00')
    date_range = [initial_start_date, initial_end_date]

    training_strategies = ['Train-test split', '1 month', '6 months', '1 year',
                           '1 month year before', 'All months before']
    model_performance_results = pd.DataFrame(columns=training_strategies)
    N_clicks = 0
    busy_computing = False
    output_text = 'Choose the model and training strategy you want and click Run to train the model.'
    number_of_dots = 0

    ml_app = JupyterDash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

    ml_app.layout = dbc.Container([
                html.Div([
                    dcc.Tabs(id='tabs', value='baseline', children=[
                        dcc.Tab(label='Baseline', value='baseline'),
                        dcc.Tab(label='Support Vector Regressor', value='SVM'),
                        dcc.Tab(label='Random Forest Regressor', value='RandomForest'),
                    ]),
                    html.Div(id='output'),
                    # html.Div(id='computing_status'),
                    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
                    html.Div(id='tabs-example-content')
                    ])
                ])

    @ml_app.callback(Output('tabs-example-content', 'children'),
                  [Input('tabs', 'value')])
    def render_content(method):
        method_param_dict = get_method_param_layout(method)
        test_feats = new_feats['2008-01-01':].copy()
        comp_dict = get_output_layout(method, test_feats, date_range[0], date_range[1], target)

        header = get_method_name(method)

        length = 4
        if method == 'baseline':
            length = 4

        #if model_performance_results.shape[0] > 0:
        df = model_performance_results.reset_index().rename(columns={'index': 'Model'})
        #else:
        #    df = pd.DataFrame({'Model': [np.nan], '': [np.nan]})


        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=False)

        layout = html.Div([html.H3(header),
                           html.Div([dbc.Row([dbc.Col([html.Div([construct_parameter_layout(method_param_dict, method)],
                                                               id='params'),
                                                       # html.Div(construct_strategy_layout(method)),
                                                       #html.Div([dbc.Checkbox(id='normalize-checkbox',
                                                        #                      key='Normalize', checked=True,
                                                        #                      className="form-check-input"),
                                                        #         dbc.Label("Normalize before training the model?",
                                                        #                   html_for="normalize-checkbox",
                                                        #                   className="form-check-label")]),
                                                       dbc.Button('Run', id='run_button', className='mr-2')],
                                                      width=3, lg=length),
                                              dbc.Col([html.Div([construct_layout(comp_dict)])], width=True, lg=8),
                                             ]),

                                     html.Br(),
                                     html.H6('Performance of tested models (Mean Absolute Error in Wh)'),
                                     html.Div([dbc.Row([table])], id='output-table')
                                    ])
                          ])
        return layout

    @ml_app.callback(Output('output', 'children'),
                     [Input('interval-component', 'n_intervals')])
    def update_computing_status(n):
        global busy_computing, output_text, number_of_dots
        # return dbc.Alert([f'{n} - {N_clicks}'], color='primary')
        #if 'Fitted' in output['props']['children'][0]:
        max_number_of_dots = 5
        number_of_dots = (number_of_dots + 1) % (max_number_of_dots + 1)
        #number_of_dots = n % (max_number_of_dots + 1)

        suffix = ''.join(['.' for _ in range(number_of_dots)])
        if busy_computing:
            output_text = f'Computing{suffix}'

        #else:
        # print(n, n_dummy)
        '''
        if n is None:
            text = 'Status: waiting...'
        else:
            if n != n_dummy:
                text = 'Status: computing...'
            else:
                text = 'Status: done computing'
        '''

        return dbc.Alert([output_text], color='primary')
        #else:
        #    return dbc.Alert(['No computing yet'], color='primary')


    @ml_app.callback([Output('comparison_plot', 'figure'), Output('output-table', 'children')],
                     [Input('run_button', 'n_clicks')],
                     [State('params', 'children'), State('tabs', 'value'), State('comparison_plot', 'figure'),
                      State('output-table', 'children'),
                      State('strategy', 'value'), State('normalize-checkbox', 'checked')])
    def run_method(n, params, method, original_figure, table_output,
                   train_strategy, normalize):
        global N_clicks, initial_start_date, initial_end_date, busy_computing, output_text, number_of_dots
        number_of_dots = 0

        try:

            if n is not None:
                busy_computing = True
                parameters = get_method_parameters(params)
                regr = get_regressor(method, **parameters)
                index = '-'.join([str(elem) for elem in [get_method_name(method), *parameters.values()]])
                if not method == 'baseline':
                    if normalize:
                        index += '-Normalized'
                    else:
                        index += '-Not normalized'
                if n > N_clicks:
                    # Check to see if 'Run' was clicked. No need to retrain the model if not
                    y_test_pred, y_test = get_predictions(regr, train_strategy, new_feats, normalize)
                    MAE = mean_absolute_error(y_test, y_test_pred)

                    y1 = pd.Series(y_test, name='True value')#.loc[start_date:end_date]
                    y2 = pd.Series(y_test_pred, name='prediction')#.loc[start_date:end_date]

                    new_fig = vp.plot_multiple_time_series([y1, y2], show_time_slider=True)
                    # new_fig.update_xaxes(range=[start_date, end_date])

                    model_performance_results.loc[index, train_strategy] = round(MAE, 2)

                    N_clicks = n

                else:
                    new_fig = go.Figure(original_figure)
                    MAE = model_performance_results.loc[index, train_strategy]

                # new_fig.update_xaxes(range=[start_date, end_date])
                output_text = f'Fitted the model (\"{index}\"). Mean absolute error: {MAE: .2f} Wh'

                # busy_computing = False
                #return dbc.Alert([text], color='primary'), new_fig
            else:
                N_clicks = 0
                original_figure['layout']['xaxis']['rangeslider']['yaxis'] = {'rangemode': 'match'}
                new_fig = go.Figure(original_figure)

                output_text = 'Choose the model and training strategy you want and click Run to train the model.'
                # done_computing = True


            #if model_performance_results.shape[0] > 0:
            df = model_performance_results.reset_index().rename(columns={'index': 'Model'})
            #else:
            #    df = pd.DataFrame({'Model': [np.nan], '': [np.nan]})

            table_output = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

            new_date_range = original_figure['layout']['xaxis']['range']
            date_range[0] = new_date_range[0]
            date_range[1] = new_date_range[1]

            new_fig['layout']['xaxis'].update(range=date_range)
            # new_fig.update_xaxes(range=[start_date, end_date])
            new_fig.update_layout(yaxis_title="Consumption [Wh]")

            busy_computing = False

            return new_fig, table_output #dbc.Alert([], color='primary'), original_figure
        except Exception as e:
            df = model_performance_results.reset_index().rename(columns={'index': 'Model'})
            table_output = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
            output_text = f'Method not initialised properly, error {e}'
            return original_figure, table_output#pd.DataFrame()

    ml_app.run_server(mode=mode, port=port, debug=debug, height=height, host=host)
