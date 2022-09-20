# ©, 2019, Sirris
# owner: HCAB

"""Collection of wrapper functions for plotly plots"""

import numpy as np
import pandas as pd
import seaborn as sb
from copy import deepcopy
from starterkits.visualization import vis_plotly_tools as vpt
from starterkits.visualization import vis_tools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)

pio.templates.default = "plotly_white"


def plot_multiple_time_series(series, xlabel='Date', ylabel='', title='',
                              trace_specs=None, series_names=None,
                              same_plot=True, show_time_slider=False,
                              show_time_frame=False, show_legend=True,
                              scatter_kwargs=None,
                              kwargs_subplots=None, mode='lines'):
    """
    Plot multiple time series in a single plot or in different subplots

    :param series: pandas series or list of pandas series
    :param xlabel: str. Label for x-axis
    :param ylabel: str. Label for y-axis
    :param title: str. plot title
    :param trace_specs: None. dictionary with trace parameters. see
    vpt.define_figure_specs for more details
    :param series_names: str or list. Name of list of names for each series.
    Default: None
    :param same_plot: bool. Whether to plot all series in the same plot or in
    different subplots. Default: True
    :param show_time_slider: bool. Whether to show the date slider underneath.
    Default: False
    :param show_time_frame: bool.  Whether to show the time frame buttons.
    Default: False
    :param show_legend: bool.  Whether to show figure legend. Default: True
    :param mode: str or list. Type of plot to make. Default: lines (warning,
    can become slow for large datasets if 'markers' are specified)
    :param kwargs_subplots. dict. Dictionary with further parameters to be
    passed to make_subplots. Default: None (empty dict)
    :param scatter_kwargs: dict. Further arguments to go.Scatter (e.g.,
    line/marker, etc). Default: None
    :return: plotly fig object
    """
    trace_specs = {} if trace_specs is None else trace_specs
    kwargs_subplots = {} if kwargs_subplots is None else kwargs_subplots
    series, series_names, trace_specs, trace_cols = \
        vpt.define_figure_specs(series, series_names, trace_specs,
                                ['alpha', 'width', 'dash', 'color', 'modes'])
    n_series = len(series)
    mode = (mode if isinstance(mode, tuple([list, np.ndarray]))
            else [mode] * n_series)
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    trace_specs['scatter_kwargs'] = scatter_kwargs
    plot_assistant = {'series': series, 'series_names': series_names,
                      'trace_cols': trace_cols, 'trace_specs': trace_specs,
                      'mode': mode,
                      'showlegend': show_legend}

    if same_plot:
        fig = go.Figure([_make_timeseries(k, plot_assistant)
                         for k in np.arange(n_series)])
        fig.update_layout(xaxis=vpt.add_date_slider(show_time_slider,
                                                    show_time_frame),
                          yaxis={'fixedrange': False},
                          xaxis_title=xlabel,
                          yaxis_title=ylabel,
                          title=title,
                          font={'size': 18, 'color': "#7f7f7f"},
                          showlegend=show_legend)
    else:
        fig = _make_subplots(series_names,
                             kwargs_subplots,
                             plot_assistant,
                             _make_timeseries)
    return fig


def _make_timeseries(k, pa):
    return go.Scatter(x=pa['series'][k].index,
                      y=pa['series'][k],
                      name=pa['series_names'][k],
                      line={c: pa['trace_specs'][c][k]
                            for c in pa['trace_cols']},
                      mode=pa['mode'][k],
                      **pa['trace_specs']['scatter_kwargs'])


def _make_subplots(series_names, kwargs_subplots, pa, plotter):
    kwargs_subplots['rows'], kwargs_subplots['cols'], walker = (
        vpt.fig_layout(len(series_names),
                       kwargs_subplots.get('rows', 1),
                       kwargs_subplots.get('cols', len(series_names))))
    fig = make_subplots(subplot_titles=series_names, **kwargs_subplots)
    for k in np.arange(len(series_names)):
        loc = walker.locate(k)
        fig.add_trace(plotter(k, pa),
                      row=int(loc[0]),
                      col=int(loc[1]))
    fig.update_layout(showlegend=pa['showlegend'])

    return fig


def plot_multiple_scatter(df, x, y,
                          color=None, size=None,
                          xlabel='', ylabel='', title='',
                          trace_specs=None, df_names=None, scatter_kwargs=None,
                          same_plot=True, kwargs_subplots=None,
                          show_legend=True, colorbar_title=None):
    """
    Plot multiple scatters in a single plot or in different subplots

    :param df: pandas dataframe or list of pandas dataframe
    :param x: str. Name of variable on the x-axis
    :param y: str. Name of variable on the y-axis
    :param color: str. Name of variable for the color scale. Default None
    :param size: str. Name of variable for the size scale. Default None
    :param xlabel: str. Label for x-axis
    :param ylabel: str. Label for y-axis
    :param title: str. plot title
    :param trace_specs: dict. dictionary with trace parameters
    :param df_names: str or list. Name of list of names for each series.
    Default: None
    :param scatter_kwargs: dict. Further arguments to go.Scatter (e.g.,
    line/marker, etc). Default: None
    :param same_plot: bool. Whether to plot all series in the same plot or in
    different subplots. Default: True
    :param kwargs_subplots: dict. Dictionary with further parameters to be
    passed to make_subplots. Default: None (empty dict)
    :param show_legend: bool.  Whether to show figure legend. Default: True
    :param colorbar_title: str. Title of colorbar. Default: None
    :return: plotly fig object
    """
    trace_specs = {} if trace_specs is None else trace_specs
    kwargs_subplots = {} if kwargs_subplots is None else kwargs_subplots
    df_names = [''] * len(vpt.make_list(df)) if df_names is None else df_names

    df, df_names, trace_specs, trace_cols = \
        vpt.define_figure_specs(df, df_names, trace_specs,
                                ['size', 'color', 'sizeref'])

    def get_size_ref(vmax, target=2):
        return [2 * np.max(vmax) / (target ** 2)] * len(vmax)
    if size is not None:
        trace_specs['sizeref'] = get_size_ref([d[size].max() for d in df],
                                              trace_specs['size'][0])
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    trace_specs['scatter_kwargs'] = scatter_kwargs
    trace_specs['scatter_kwargs']['mode'] = (trace_specs['scatter_kwargs']
                                             .get('mode', 'markers'))
    trace_specs['colorbar'] = trace_specs.get(
        'colorbar', {'title': colorbar_title or color})
    n_df = len(df)
    trace_specs['showscale'] = [True if (color is not None) & (len(df) == 1)
                                else False] * len(df)

    plot_assistant = {'df': df, 'df_names': df_names,
                      'x': x, 'y': y, 'size': size, 'color': color,
                      'trace_cols': trace_cols, 'trace_specs': trace_specs,
                      'showscale': True if color is not None else False,
                      'showlegend': show_legend}

    if same_plot:
        fig = go.Figure([_make_scatter(k, plot_assistant)
                         for k in np.arange(n_df)])
        fig.update_layout(xaxis_title=xlabel,
                          yaxis_title=ylabel,
                          title=title,
                          font={'size': 18, 'color': "#7f7f7f"},
                          showlegend=show_legend)
    else:
        fig = _make_subplots(df_names,
                             kwargs_subplots,
                             plot_assistant,
                             _make_scatter)

    return fig


def _make_scatter(k, pa):
    trace_specs = deepcopy(pa['trace_specs'])
    if 'marker' in trace_specs['scatter_kwargs'].keys():
        marker = trace_specs['scatter_kwargs']['marker']
        trace_specs['scatter_kwargs'].pop('marker')
    else:
        marker = {}
    if 'hovertext' in trace_specs['scatter_kwargs'].keys():
        hovertext = trace_specs['scatter_kwargs']['hovertext']
        trace_specs['scatter_kwargs'].pop('hovertext')
    else:
        hovertext = ''
    for c in ['sizeref', 'showscale']:
        marker.update({c: trace_specs[c][k]})

    marker['colorbar'] = marker.get('colorbar', trace_specs['colorbar'])
    marker['size'] = marker.get('size',
                                trace_specs['size'][k]
                                if pa['size'] is None
                                else pa['df'][k][pa['size']])
    marker['color'] = marker.get('color',
                                 trace_specs['color'][k]
                                 if pa['color'] is None
                                 else pa['df'][k][pa['color']])

    return go.Scattergl(
        x=pa['df'][k][pa['x']],
        y=pa['df'][k][pa['y']],
        marker=marker,
        hovertext=hovertext,
        name=pa['df_names'][k],
        **trace_specs['scatter_kwargs'])


def histogram(series, xlabel='', ylabel='', title='',
              trace_specs=None, series_names=None,
              same_plot=True,
              hist_specs=None, bins=None,
              kwargs_subplots=None):
    """
    Make plotly histogram

    :param series: pandas series or list of pandas series
    :param xlabel: str. Label for x-axis
    :param ylabel: str. Label for y-axis
    :param title: str. plot title
    :param trace_specs: dict. dictionary with trace parameters
    :param series_names: str or list. Name of list of names for each series.
    Default: None
    :param same_plot: bool. Whether to plot all series in the same plot or in
    different subplots. Default: True
    :param hist_specs: dict. Dictionary with hist-specific options (e.g.
    barmode, histnorm, etc)
    :param bins: int, list, array. Bins to use for the histogram. Can be single
    number, list of ints, etc.
    Default: None (will estimate number of bins from sturges formula
    :param kwargs_subplots: dict. Dictionary with further parameters to be
    passed to make_subplots. Default: None (empty dict)

    returns: fig: plotly object
    """
    trace_specs = {} if trace_specs is None else trace_specs
    hist_specs = {} if hist_specs is None else hist_specs
    kwargs_subplots = {} if kwargs_subplots is None else kwargs_subplots

    # barmode: stack, group, relative, overlay
    hist_specs['barmode'] = hist_specs.get('barmode', 'overlay')
    # histnorm: percent, probability, density
    hist_specs['histnorm'] = hist_specs.get('histnorm', '')
    hist_specs['cumulative_enabled'] = hist_specs.get('cumulative_enabled',
                                                      False)

    series = vpt.make_list(series)
    series_names = ([s.name for s in series]
                    if series_names is None
                    else series_names)

    # check data type
    isnumeric = [pd.api.types.is_numeric_dtype(s) for s in series]

    bins = [bins] * len(series)

    def get_bins(b, x):
        st = None
        en = None
        if b is None:
            nbins = 1 + 3.322 * np.log(len(x))
        elif isinstance(b, int):
            nbins = b
        elif isinstance(b, (list, np.ndarray)):
            nbins = len(b)
            st = np.min(b)
            en = np.max(b)
        else:
            nbins = None

        if (st is None) | (en is None):
            st, en = x.agg([min, max])
            nbins = nbins or 20

        step = np.diff([st, en])[0] / nbins
        st = st - step
        en = en + step

        return {'start': st, 'end': en, 'size': step}

    bins = [get_bins(b, series[k]) for k, b in enumerate(bins) if isnumeric[k]]

    series, series_names, trace_specs, trace_cols = \
        vpt.define_figure_specs(series, series_names, trace_specs,
                                ['color', 'alpha'])
    plot_assistant = {'series': series, 'series_names': series_names,
                      'bins': bins, 'hist_specs': hist_specs,
                      'trace_specs': trace_specs,
                      'isnumeric': isnumeric}
    n_series = len(series)
    if same_plot:
        fig = go.Figure([_make_histogram(k, plot_assistant)
                         for k in np.arange(n_series)])
        fig.update_layout(barmode=hist_specs['barmode'],
                          xaxis_title=xlabel,
                          yaxis_title=ylabel,
                          title=title,
                          font={'size': 18, 'color': "#7f7f7f"})
        fig.update_layout(showlegend=False if len(series) == 1 else True)
    else:
        fig = _make_subplots(series_names,
                             kwargs_subplots,
                             plot_assistant,
                             _make_histogram)

    return fig


def _make_countplot(k, pa):
    cts = pa['series'][k].value_counts()
    return go.Bar(x=cts.index.astype(str),
                  y=cts,
                  name=pa['series_names'][k],
                  marker={'color': pa['trace_specs']['color'][k]})


def _make_histogram(k, pa):
    if pa['isnumeric'][k]:
        return go.Histogram(x=pa['series'][k],
                            name=pa['series_names'][k],
                            xbins=pa['bins'][k],
                            histnorm=pa['hist_specs']['histnorm'],
                            cumulative_enabled=(pa['hist_specs']
                                                ['cumulative_enabled']),
                            marker={'color': pa['trace_specs']['color'][k]})
    else:
        return _make_countplot(k, pa)


def boxplot(series,
            xlabel='', ylabel='', title='',
            trace_specs=None, series_names=None,
            same_plot=True, kwargs_subplots=None,
            box_specs=None, orient='vertical'):
    """
    Make plotly boxplot

    :param series: pandas series or list of pandas series
    :param xlabel: str. Label for x-axis
    :param ylabel: str. Label for y-axis
    :param title: str. plot title
    :param trace_specs: dict. dictionary with trace parameters
    :param series_names: str or list. Name of list of names for each series.
    Default: None
    :param same_plot: bool. Whether to plot all series in the same plot or in
    different subplots. Default: True
    :param box_specs: dict. Dictionary with boxplot-specific options (e.g.
    notched, boxpoints, etc)
    :param orient: str, Orientation of the boxplots ('vertical', 'horizontal'
    or 'v', 'h', or 'vert', 'horiz', ...). Default: 'vertical
    :param kwargs_subplots: dict. Dictionary with further parameters to be
    passed to make_subplots. Default: None (empty dict)

    returns: fig: plotly object
    """
    trace_specs = {} if trace_specs is None else trace_specs
    box_specs = {} if box_specs is None else box_specs
    kwargs_subplots = {} if kwargs_subplots is None else kwargs_subplots

    box_specs['notched'] = box_specs.get('notched', True)
    box_specs['boxpoints'] = box_specs.get('boxpoints', False)  # can also be
    # outliers, or suspectedoutliers, or False
    box_specs['jitter'] = box_specs.get('jitter', 0.3)
    box_specs['pointpos'] = box_specs.get('pointpos', 0)
    box_specs['boxmean'] = box_specs.get('boxmean', True)

    series = vpt.make_list(series)
    series_names = ([s.name for s in series]
                    if series_names is None
                    else series_names)

    series, series_names, trace_specs, _ = \
        vpt.define_figure_specs(series, series_names, trace_specs,
                                ['color', 'alpha'])
    plot_assistant = {'series': series, 'series_names': series_names,
                      'box_specs': box_specs, 'trace_specs': trace_specs,
                      'orient': 'x' if orient.startswith('h') else 'y'}
    n_series = len(series)
    if same_plot:
        fig = go.Figure([_make_boxplot(k, plot_assistant)
                         for k in np.arange(n_series)])
        fig.update_layout(xaxis_title=xlabel,
                          yaxis_title=ylabel,
                          title=title,
                          font={'size': 18, 'color': "#7f7f7f"})
        fig.update_layout(showlegend=False if len(series) == 1 else True)
    else:
        fig = _make_subplots(series_names,
                             kwargs_subplots,
                             plot_assistant,
                             _make_boxplot)

    return fig


def _make_boxplot(k, pa):
    return go.Box({pa['orient']: pa['series'][k],
                   'name': pa['series_names'][k],
                   'marker': {'color': pa['trace_specs']['color'][k]},
                   **pa['box_specs']})


def plot_scatter_matrix(ds,
                        grouping=None,
                        columns=None,
                        cmap='Set1',
                        alpha=0.6):
    """
    Build scatter matrix

    :param ds: dataframe. input dataframe
    :param grouping: str. variable to group data. Default: None (no grouping)
    :param columns: list/array. List of columns in dataset to use in the
    scatter matrix. Default: None (use all
    numerical columns)
    :param cmap: str. Colormap to use. Default: 'Set1'
    :param alpha: float. Transparency level for the markers. Default: 0.6
    :return: fig: plotly object
    """
    columns = (ds.select_dtypes('number').columns
               if columns is None else columns)
    color = (sb.xkcd_rgb['cool blue']
             if grouping is None
             else ds[grouping].astype('category').cat.codes)
    if grouping is not None:
        cmap = list(vis_tools.define_color_categories(ds[grouping].unique(),
                                                      cmap).values())
    else:
        cmap = [sb.xkcd_rgb['cool blue']]
    cmap = [vpt.to_rgba(c, alpha) for c in cmap]
    fig = go.Figure(data=go.Splom(dimensions=[{'label': vis_tools.prettify(c),
                                               'values': ds[c]}
                                              for c in columns],
                                  text=(None
                                        if grouping is None
                                        else ds[grouping]),
                                  marker={'color': color,
                                          'showscale': False,  # colors
                                          # encode categorical variables
                                          'line_color': 'white',
                                          'line_width': 0.5,
                                          'colorscale': cmap}))
    fig.update_layout(dragmode='select', hovermode='closest',
                      width=600, height=600)

    return fig


def plot_heatmap(ds,
                 rows=None,
                 cols=None,
                 cmap='Viridis',
                 xlabel='', ylabel='', title=''):
    """
    Build heatmap. Row na

    :param ds: dataframe. input dataframe
    :param rows: arrray. Row vector. Default: None (use dataframe index)
    :param cols: arrray. Column vector. Default: None (use dataframe columns)
    :param cmap: str. Colormap to use. Default: 'Set1'
    :param xlabel: str. Label for x-axis
    :param ylabel: str. Label for y-axis
    :param title: str. plot title
    :return: fig: plotly object
    """
    rows = ds.index if rows is None else rows
    cols = ds.columns if cols is None else cols
    fig = go.Figure(data=go.Heatmap(z=ds,
                                    x=cols,
                                    y=rows,
                                    colorscale=cmap))

    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=ylabel,
                      title=title,
                      font={'size': 18, 'color': "#7f7f7f"})

    return fig
