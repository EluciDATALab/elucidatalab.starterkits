# ©, 2019, Sirris
# owner: HCAB

"""functions facilitating the use of widgets in plotly plots"""

import pandas as pd
import numpy as np
from copy import deepcopy
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from starterkits.visualization import vis_tools
from starterkits.visualization import vis_plotly as vp
from starterkits.visualization import vis_plotly_tools as vpt
from starterkits.visualization import vis_plotly_widgets_tools as vpwt

import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)


def timeseries_with_variable_selector(df,
                                      groupby=None,
                                      columns=None,
                                      select_multiple=True,
                                      description='Variable',
                                      default=None,
                                      kwargs_ts=None,
                                      kwargs_layout=None,
                                      aggfun=np.median):
    """
    Plot time series with widget to select variable

    :param df: dataframe. dataset to use. Must have datetime index
    :param groupby: str. Group series by this variable. Default: None (no
    grouping)
    :param columns: list/array. columns to make available for widget. Default:
    None (use all numeric columns)
    :param select_multiple: bool. Whether to allow selection of multiple items
    in columns.
    Default: True
    :param description: str. Name of variable selector widget
    :param default: str. Variable that is selected by default. Default: None
    (use first available)
    :param kwargs_ts: dict. Further parameters to be passed to
    vp.plot_multiple_time_series. Default: None (no params)
    :param kwargs_layout: dict. Further parameters to be passed to
    fig.update_layout. Default: None (no params)
    :param aggfun: function. If no grouping is provided, target column will be
    aggregated over each unique timestamp using this function. Default:
    np.median
    :return:
    plot_timeseries: function that makes the plot
    var_selector: widget to select variable to plot
    date_range_slider: widget with date range slider
    """
    kwargs_ts = {} if kwargs_ts is None else kwargs_ts
    kwargs_layout = {} if kwargs_layout is None else kwargs_layout

    columns = (list(np.setdiff1d(df.select_dtypes('number').columns, groupby))
               if columns is None else columns)
    columns = [(vis_tools.prettify(c), c) for c in columns]
    default = columns[0][1] if default is None else default
    # define selector
    select_widget = (widgets.SelectMultiple if select_multiple
                     else widgets.Dropdown)
    default = vpt.make_list(default) if select_multiple else default
    var_selector = select_widget(options=columns,
                                 description=description,
                                 value=default)

    date_range_slider = vpwt.define_slider_date_range(df.index.min(),
                                                      df.index.max())

    def plot_timeseries(cols, period=None):
        ts_kwargs = deepcopy(kwargs_ts)
        df0 = df.copy()
        cols = vpt.make_list(cols) if select_multiple is False else cols
        if period is not None:
            df0 = df0.loc[period[0]: period[1]]
        if groupby:
            cols = cols[0]
            groups = df0[groupby].unique()
            fig = vp.plot_multiple_time_series(
                [df0.loc[df0[groupby] == g, cols] for g in groups],
                series_names=groups, show_time_slider=False,
                title=vis_tools.prettify(cols),
                **ts_kwargs)
            if 'yaxis_title' in kwargs_layout.keys():
                fig.update_layout(
                    yaxis_title=kwargs_layout['yaxis_title'][cols])
        else:
            fig = vp.plot_multiple_time_series(
                [df0.groupby(df0.index)[c].agg(aggfun) for c in cols],
                series_names=[vis_tools.prettify(c) for c in cols],
                same_plot=False, **ts_kwargs)
            vpt.update_labels_subplots(fig, kwargs_layout, cols)

        vpwt.update_layout(fig, kwargs_ts, kwargs_layout)
        plt.close()
        display(fig)

    return plot_timeseries, var_selector, date_range_slider


def plot_transform_time_series(df, methods, controllers=None, columns=None,
                               plot_original=True, add_shape=None,
                               kwargs_method=None, kwargs_ts=None,
                               kwargs_layout=None):
    """
    Apply one of possible methods to a time series and display the transformed

    time series.

    :param df: dataframe. Main dataset. Must have datetime index
    :param methods: dict. Dictionary of possible methods
    :param controllers: list. List of widget controls. Each element is a
    dictionary that will be parsed by vpwt.get_controller function to define a
    widget. Default: None (no controller)
    :param columns: str/dict. Either the name of the variable to be transformed,
    either a dictionary to be passed to vpwt.get_controller to be parsed and
    build a widget with variable selector. Default: None (use all available
    numeric columns)
    :param plot_original: bool. Whether to plot the original plot. Default: True
    :param add_shape: function. Function to add shapes to a plotly figure. It
    must accept three input parameters: the figure object, the transformed
    dataset and the column selector. See template notebook for further details.
    Default: None (no shape to add)
    :param kwargs_method: dict. Dictionary to be parsed into a widget using
    vpwt.get_controller to define the methods widget. Default: None (use
    defaults)
    :param kwargs_ts: dict. Further parameters to be passed to
    vp.plot_multiple_time_series. Default: None (no params)
    :param kwargs_layout: dict. Further parameters to be passed to
    fig.update_layout. Default: None (no params)
    :return:
    transform_plot: transformed dataset
    method_picker: widget with method selector
    controllers: list of widgets that control variable inputs to methods
    """
    columns = columns if columns is not None else {}
    kwargs_method = kwargs_method if kwargs_method is not None else {}
    # DEFINE WIDGETS
    if controllers is not None:
        controllers = [vpwt.get_controller(c) for c in controllers]
    else:
        controllers = []

    # dropdown method for smoothing method
    kwargs_method['options'] = kwargs_method.get('options',
                                                 list(methods.keys()))
    method_picker = widgets.Dropdown(**kwargs_method)

    if not isinstance(columns, str):
        columns['widget'] = columns.get('widget', 'SelectMultiple')
        col_options = [(vis_tools.prettify(c), c)
                       for c in df.select_dtypes('number').columns.tolist()]
        columns['options'] = columns.get('options', col_options)
        default_value = columns['options'][0][1]
        default_value = (default_value[-1] if isinstance(default_value, tuple)
                         else default_value)
        default_value = ([default_value]
                         if columns['widget'] == 'SelectMultiple'
                         else default_value)
        columns['value'] = columns.get('value', default_value)
        controllers = [vpwt.get_controller(columns)] + controllers

    # DEFINE PLOTTING FUNCTION
    def transform_df(dft, method, **args):
        args = list(args.values())
        if isinstance(columns, str):
            column = [columns]
        else:
            column = [args[0]] if isinstance(args[0], str) else args[0]
            args = args[1:]
        df_out = []
        for c in column:
            # if window is None:
            df_out0 = methods[method](
                dft,
                c,
                *[a for a in args if a is not None])
            df_out0 = (df_out0.to_frame() if isinstance(df_out0, pd.Series)
                       else df_out0)
            if f'{c}_Original' not in df_out0.columns:
                df_out0[f'{c}_Original'] = df[c]
            df_out.append(df_out0)
        return df_out, column

    def make_plot(df_out, plt_cols):
        ts_kwargs = {} if kwargs_ts is None else deepcopy(kwargs_ts)
        layout_kwargs = {} if kwargs_layout is None else deepcopy(kwargs_layout)
        series = []
        series_names = []
        for k, c in enumerate(plt_cols):
            cols = [f'{c}_Original', c] if plot_original else [c]
            series = series + [df_out[k][c0] for c0 in cols]
            series_names = series_names + cols
        trace_specs = ts_kwargs.get('trace_specs', {})
        ts_default_kwargs = {
            'alpha': trace_specs.get('alpha',
                                     [0.75, 1] if plot_original else [1]),
            'width': trace_specs.get('width',
                                     [1, 3] if plot_original else [2]),
            'cmap': trace_specs.get('cmap',
                                    'Paired' if plot_original else 'Set1')}

        ts_kwargs['trace_specs'] = ts_kwargs.get('trace_specs', {})
        ts_kwargs['trace_specs']['alpha'] = ts_kwargs['trace_specs'].get(
            'alpha', ts_default_kwargs['alpha'] * len(plt_cols))
        ts_kwargs['trace_specs']['width'] = ts_kwargs['trace_specs'].get(
            'width', ts_default_kwargs['width'] * len(plt_cols))
        ts_kwargs['trace_specs']['cmap'] = ts_kwargs['trace_specs'].get(
            'cmap', ts_default_kwargs['cmap'])

        fig = vp.plot_multiple_time_series(series,
                                           series_names=[vis_tools.prettify(s)
                                                         for s in series_names],
                                           **ts_kwargs)
        if add_shape is not None:
            add_shape(fig, df_out, plt_cols)

        vpwt.update_layout(fig, ts_kwargs, layout_kwargs)
        plt.close()
        display(fig)

    def transform_plot(method, **args):
        df_out, plotting_columns = transform_df(df, method, **args)
        make_plot(df_out, plotting_columns)

    return transform_plot, method_picker, controllers


def plot_distribution(df,
                      groupby=None,
                      columns=None,
                      kwargs_dist=None,
                      trace_specs=None):
    """
    Plot a boxplot or histogram distribution of a variable

    :param df: dataframe. Main dataset
    :param groupby: str. Group series by this variable. Default: None (no
    grouping)
    :param columns: list/array. columns to make available for widget. Default:
    None (use all numeric columns)
    :param kwargs_dist: dict. Further arguments to be passed to vp.boxplot or
    vp.histogram. Default: None (no arguments)
    :param trace_specs: dict. Further arguments to control aesthetics of plot
    trace. Default: None (use defaults)
    :return:
    plot_dist: plotting function
    columns: widget that controls variable to plot
    plot_type: widget that controls plot type
    """
    kwargs_dist = {} if kwargs_dist is None else kwargs_dist
    columns = {} if columns is None else columns
    trace_specs = {} if trace_specs is None else trace_specs

    columns['widget'] = columns.get('widget', 'Select')
    columns['description'] = columns.get('description', 'Variable')
    columns['options'] = columns.get('options', (df
                                                 .select_dtypes('number')
                                                 .columns
                                                 .tolist()))
    default_value = columns['options'][0]
    default_value = ([default_value] if columns['widget'] == 'SelectMultiple'
                     else default_value)
    columns['value'] = columns.get('value', default_value)
    columns = vpwt.get_controller(columns)

    plot_type = widgets.ToggleButtons(options=['Boxplot', 'Histogram'],
                                      description='Plot type')

    dist_keys = {'Boxplot': ['notched', 'boxpoints', 'jitter',
                             'pointpos', 'boxmean'],
                 'Histogram': ['barmode', 'histnorm', 'cumulative_enabled']}

    def plot_dist(col, plt_type):

        series = (df[col]
                  if groupby is None else [df.loc[df[groupby] == i, col]
                                           for i in df[groupby].unique()])
        series_names = col if groupby is None else df[groupby].unique()

        # plt_method = vp.boxplot if plt_type == 'Boxplot' else vp.histogram
        if len(kwargs_dist) > 0:
            kwargs_plt = {k: v for k, v in kwargs_dist.items()
                          if k in dist_keys[plt_type]}
        else:
            kwargs_plt = None
        if plt_type == 'Boxplot':
            orient = (None if len(kwargs_dist) == 0
                      else kwargs_dist.get('orient', None))
            fig = vp.boxplot(series, series_names=series_names, orient=orient,
                             same_plot=kwargs_dist.get('same_plot', True),
                             box_specs=kwargs_plt, trace_specs=trace_specs)
        else:
            bins = (None if len(kwargs_dist) == 0
                    else kwargs_dist.get('bins', None))
            fig = vp.histogram(series, series_names=series_names, bins=bins,
                               same_plot=kwargs_dist.get('same_plot', True),
                               hist_specs=kwargs_plt, trace_specs=trace_specs)

        fig.show()

    return plot_dist, columns, plot_type


def plot_scatterplot(df,
                     groupby=None,
                     columns=None,
                     kwargs_sc=None,
                     kwargs_layout=None,
                     allow_color=True,
                     allow_size=True):
    """
    Plotting function to make interactive scatterplot

    :param df: dataframe. Main dataset
    :param groupby: str. Group series by this variable. Default: None (no
    grouping)
    :param columns: list/array. columns to make available for widget. Default:
    None (use all numeric columns)
    :param kwargs_sc: dict. Further arguments to be passed to
    vp.plot_multiple_scatterplots. Default: None (no arguments)
    :param kwargs_layout: dict. Further parameters to be passed to
    fig.update_layout. Default: None (no params)
    :param allow_color: bool. Whether to add widget selector for color
    dimension. Default: True
    :param allow_size: bool. Whether to add widget selector for size dimension.
    Default: True
    :return:
    plot_scatter: plotting function
    col_controller: list of widgets for variable selection per dimension
    limit_samples: widget controlling number of points to plot
    same_plot: widger to define whether to plot all scatters in same plot.
    Default: True
    """
    kwargs_layout = {} if kwargs_layout is None else kwargs_layout
    columns = {} if columns is None else columns
    kwargs_sc = {} if kwargs_sc is None else kwargs_sc

    columns['widget'] = columns.get('widget', 'Dropdown')
    col_options = columns.get('options', (df
                                          .select_dtypes('number')
                                          .columns
                                          .tolist()))

    col_controller = []
    dims = ['X-axis', 'Y-axis']
    dims = dims + (['Color'] if allow_color else [])
    dims = dims + (['Size'] if allow_size else [])
    for k, c in enumerate(dims):
        add_it = [] if c.endswith('axis') else [None]
        columns['options'] = add_it + col_options
        default_value = columns['options'][k if c.endswith('axis') else 0]
        columns['value'] = default_value
        columns['description'] = c
        col_controller.append(vpwt.get_controller(columns))

    limit_samples = widgets.SelectionSlider(
        description='Limit number of samples',
        options=['1000', '10000', '100000', '1000000', 'All'],
        value='1000')
    same_plot = widgets.Checkbox(description='Same plot', value=True)

    def plot_scatter(xlab, ylab, limit, same_plt, **args):
        args = list(args.values())
        color = args[0] if allow_color else None
        size = args[-1] if allow_size else None
        if limit != 'All':
            n = int(limit)
            if n > len(df):
                df_sub = df
            else:
                df_sub = df.sample(n)
        else:
            df_sub = df
        if size is not None:
            df_sub[size] = df_sub[size] + np.abs(df_sub[size].min())
        df_plot = (df_sub if groupby is None
                   else [df_sub.loc[df_sub[groupby] == i]
                         for i in df_sub[groupby].unique()])
        df_names = '' if groupby is None else df_sub[groupby].unique()

        fig = vp.plot_multiple_scatter(df_plot,
                                       xlab,
                                       ylab,
                                       color=color,
                                       size=size,
                                       xlabel=xlab,
                                       ylabel=ylab,
                                       df_names=df_names,
                                       same_plot=same_plt,
                                       **kwargs_sc)
        fig_layout = vpt.fig_layout(len(vpt.make_list(df_plot)))
        kwargs_layout0 = deepcopy(kwargs_layout)
        for k1, c1 in enumerate(['height', 'width']):
            if c1 in kwargs_layout0.keys():
                kwargs_layout0[c1] = kwargs_layout0[c1] if same_plot \
                    else kwargs_layout0[c1] * fig_layout[k1] * 0.75
        vpwt.update_layout(fig, kwargs_sc, kwargs_layout0)
        fig.show()

    return plot_scatter, col_controller, limit_samples, same_plot


def timeseries_with_variable_filter(df,
                                    groupby=None,
                                    columns=None,
                                    filters=None,
                                    filter_column=None,
                                    description='Variable',
                                    filter_description='ID',
                                    filter_value=None,
                                    default=None,
                                    kwargs_ts=None,
                                    kwargs_layout=None,
                                    temporal=True,
                                    time_index=None,
                                    add_median_line=False):
    """
    Plot time series with widget to select variable and filter

    :param df: dataframe. dataset to use. Must have datetime index
    :param groupby: str. Group series by this variable. Default: None (no
    grouping)
    :param filters: possible filters to apply
    :param filter_column: str, column to filter on
    :param columns: list/array. columns to make available for widget. Default:
    None (use all numeric columns)
    :param description: str. Name of variable selector widget
    :param filter_description: str, name of filter for widget (default: ID)
    :param filter_value:  selected value in Dropdown
    :param default: str. Variable that is selected by default. Default: None
    (use first available)
    :param kwargs_ts: dict. Further parameters to be passed to
    vp.plot_multiple_time_series. Default: None (no params)
    :param kwargs_layout: dict. Further parameters to be passed to
    fig.update_layout. Default: None (no params)
    :param temporal: bool, whether index is Timeindex or not
    :param time_index: str, time index name
    :param add_median_line: bool, whether to plot temporal median or not
    (default False) timestamp using this function. Default: np.median

    :return:
    plot_timeseries: function that makes the plot
    var_selector: widget to select variable to plot
    date_range_slider: widget with date range slider
    """
    kwargs_ts = {} if kwargs_ts is None else kwargs_ts
    kwargs_layout = {} if kwargs_layout is None else kwargs_layout

    columns = (np.setdiff1d(df.select_dtypes('number').columns, groupby)
               if columns is None else columns)
    default = columns[0] if default is None else default

    # define selector
    var_selector = widgets.SelectMultiple(options=columns,
                                          description=description,
                                          value=vpt.make_list(default))
    var_filter = widgets.Dropdown(options=filters,
                                  description=filter_description,
                                  value=filter_value)

    if temporal:
        date_range_slider = vpwt.define_slider_date_range(df.index.min(),
                                                          df.index.max())
    else:
        date_range_slider = widgets.IntSlider(min=df[time_index].min(),
                                              max=df[time_index].max(),
                                              value=50)

    def plot_timeseries(cols, period=None, filter_id=None):
        ts_kwargs = deepcopy(kwargs_ts)
        df0 = df.copy()
        if period is not None:
            if isinstance(period, int):
                df0 = df0[df0[filter_column] == filter_id]
                if time_index:
                    df0 = df0.set_index(time_index).sort_index()
                df0 = df0.iloc[: period, :]
                print()
            else:
                df0 = df0.loc[str(period[0]): str(period[1])]
        if groupby:
            cols = vpt.make_list(list(cols))[0]
            groups = df0[groupby].unique()
            fig = vp.plot_multiple_time_series(
                [df0.loc[df0[groupby] == g, cols] for g in groups],
                series_names=groups, show_time_slider=False,
                **ts_kwargs)

        else:
            fig = vp.plot_multiple_time_series(
                [df0[c] for c in cols],
                series_names=[vis_tools.prettify(c) for c in cols],
                same_plot=False, **ts_kwargs)
        if add_median_line:
            for i, c in enumerate(cols):
                fig.add_shape(
                    dict(type="line",
                         xref=f"x{i + 1 if i > 0 else ''}",
                         yref=f"y{i + 1 if i > 0 else ''}",
                         x0=df0.index.min(),
                         y0=df0[c].median(),
                         x1=df0.index.max(),
                         y1=df0[c].median(),
                         line=dict(
                             color="Gray",
                             width=1,
                             dash='dot')
                         )
                )
        vpwt.update_layout(fig, kwargs_ts, kwargs_layout)
        fig.show()
    return plot_timeseries, var_selector, date_range_slider, var_filter
