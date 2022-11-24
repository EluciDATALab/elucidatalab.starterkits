# ©, 2019, Sirris
# owner: HCAB

"""support functions for vis_plotly"""
import seaborn as sb
import numpy as np
import pandas as pd
from starterkits.visualization import vis_tools


def add_date_slider(show_time_slider=True, show_time_frame=True):
    """
    Returns a date slider to be added to the figure.

    Also includes time frame buttons that will show up in the top. Both are
    optional

    :param show_time_slider: bool. Whether to show the time slider.
    Default: True
    :param show_time_frame: bool. Whether to show the time frame buttons.
    Default: True
    :return:
    """
    slider = {'rangeselector':
              {'buttons': [
                  {'count': 1, 'label': "1m", 'step': "month",
                   'stepmode': "backward"},
                  {'count': 6, 'label': "6m", 'step': "month",
                   'stepmode': "backward"},
                  {'count': 1, 'label': "1y", 'step': "year",
                   'stepmode': "backward"},
                  {'step': 'all'}]},
              'rangeslider': {'visible': show_time_slider},
              'type': "date"}
    if show_time_frame is False:
        slider.pop('rangeselector')
    return slider


def make_list(x):
    """
    Always return input as a list

    :param x: list, numpy array, other
    :return: list
    """
    return x if isinstance(x, list) else list(x)\
        if isinstance(x, np.ndarray) else [x]


def fig_layout(n, nrow=None, ncol=None):
    """
    Estimate layout of a figure and return a walker object to position

    different subplots in a figure

    :param n: int. number of subplots.
    :param nrow: n rows. Default: None
    :param ncol: n cols. Default: None
    :return: nrow, int
    :return: ncol, int
    :return: walker, array with same shape as subplot layout and indicating
    its position
    """
    if nrow is None:
        nrow, ncol = vis_tools.optimal_layout(n)

    walker = Walker(nrow, ncol)
    return nrow, ncol, walker


class Walker:
    """
    This class allows looping through plotting elements arranged in a

    nrow*ncol grid
    """

    def __init__(self, rows, cols):
        """Define variables"""
        self.rows = rows
        self.cols = cols
        self.walker = self.define_walker()

    def define_walker(self):
        """Define layout"""
        return np.arange(self.rows * self.cols).reshape((self.rows, self.cols))

    def locate(self, k):
        """Locate current plot"""
        return np.int64(np.argwhere(self.walker == k)[0] + 1)


def update_labels_subplots(fig, kwargs_layout, cols):
    """
    Update layout of individual subplots

    :param fig: plotly figure object
    :param kwargs_layout: dict. fig layout arguments to be updated
    :param cols: list of columns indicating each subplot
    :return:
    """
    axis = sorted([k[1:] for k in fig.layout if k.startswith('xaxis')])
    for a in ['x', 'y']:
        if f'{a}axis_title' in kwargs_layout.keys():
            for k, c in enumerate(cols):
                fig.update_layout(
                    **{f'{a}{axis[k]}':
                       {'title': kwargs_layout[f'{a}axis_title'][c]}})


def to_rgba(c, alpha=1):
    """
    Convert an rgb fraction color numpy array to rgba str

    :param c: numpy array. rgb fraction color numpy array
    :param alpha: float. level of transparency. Default: 1
    :return:
    rgba str
    """
    return 'rgba(%d,%d,%d,%0.1f)' % tuple(np.append(np.array(c) * 255, alpha))


def get_colors(cmap, alpha):
    """
    Return list of rgba str from colormap and alpha level

    :param cmap: str. color palette from seaborn (sb.color_palette)
    :param alpha: float. level of transparency
    :return: list of rgba str
    """
    n = len(alpha)
    colors = sb.color_palette(cmap)[:n]
    colors = [to_rgba(c, alpha[k]) for k, c in enumerate(colors)]
    return colors


def define_figure_specs(series, series_names, trace_specs, trace_cols):
    """
    Define the specs for plotting using a mix of input specs and default

    specs (when input spec is not provided). It also formats the data for
    plotting (put in list, etc)

    :param series: Pandas object or list of pandas object
    :param series_names: name of each series or None (will be drawn from series)
    :param trace_specs: dictionary with trace specs
    :param trace_cols: list of str. columns to use for the trace specs
    :return: series, series_names, trace_specs, trace_cols
    """
    if isinstance(series_names, pd.Index):
        series_names = series_names.values
    series = make_list(series)
    n_series = len(series)
    series_names = ([v.name for v in series]
                    if series_names is None else make_list(series_names))

    defaults = {'alpha': trace_specs.get('alpha', 1),
                'width': trace_specs.get('width', 1),
                'dash': trace_specs.get('dash', 'solid'),
                'cmap': trace_specs.get('cmap',
                                        'Dark2' if n_series > 5 else 'Set1'),
                'size': trace_specs.get('size', 10),
                'sizeref': trace_specs.get('sizeref', 1)}

    for c in np.setdiff1d(list(defaults.keys()), ['color', 'cmap']):
        if c in trace_specs.keys():
            if len(make_list(trace_specs[c])) == 1:
                trace_specs[c] = make_list(trace_specs[c]) * n_series
        else:
            trace_specs[c] = [defaults[c]] * n_series

    if 'color' in trace_cols:
        trace_specs['color'] = trace_specs.get('color',
                                               get_colors(defaults['cmap'],
                                                          trace_specs['alpha']))
        trace_specs['color'] = make_list(trace_specs['color'])
    for k, c in enumerate(trace_specs['color']):
        if isinstance(c, (list, np.ndarray, tuple)):
            trace_specs['color'][k] = to_rgba(trace_specs['color'][k])

    trace_cols = np.setdiff1d(trace_cols, ['alpha', 'modes'])

    def size_up(x, n):
        x = make_list(x)
        if len(x) != n:
            x = x * n
        return x
    trace_specs = {c: size_up(trace_specs[c], n_series) for c in trace_cols}

    return series, series_names, trace_specs, trace_cols


def get_shape_specs(sh_specs, sh_type, xref, yref):
    """
    Prepare dictionary with shape specs from inputs

    :param sh_specs: dictionary. input sh_pecs
    :param sh_type: str. shape type
    :param xref: str. Optional: x-reference of plot
    :param yref: str. Optional: y-reference of plot
    :return:
    """
    sh_specs = {} if sh_specs is None else sh_specs
    for k in ['dash', 'width']:
        if k in sh_specs.keys():
            sh_specs[f'line_{k}'] = sh_specs.pop(k)
    if 'alpha' in sh_specs.keys():
        sh_specs['opacity'] = sh_specs.pop('alpha')

    shp_specs = {'type': sh_type, 'xref': xref, 'yref': yref}
    shp_specs.update(sh_specs)

    return shp_specs


def add_shape(fig, x0, y0, x1=None, y1=None,
              sh_specs=None, sh_type='line',
              xref=None, yref=None):
    """
    Update figure layout with shape

    :param fig: plotly figure object
    :param x0: x-coordinate of left-most point in shape
    :param y0: y-coordinate of left-most point in shape
    :param x1: x-coordinate of right-most point in shape.
    Default: None (will use same as x0)
    :param y1: y-coordinate of right-most point in shape.
    Default: None (will use same as y0)
    :param sh_specs: dictionary with shape specs. Default: {} (use defaults)
    :param sh_type: str. Type of shape('line' or 'rect'). Default: 'line'
    :param xref: str. subplot x-reference (e.g. x1). Default: None
    :param yref: str. subplot y-reference (e.g. y1). Default: None
    :return:
    """
    data = {'x0': x0, 'x1': x0 if x1 is None else x1,
            'y0': y0, 'y1': y0 if y1 is None else y1}

    def format_timestamp(x):
        return pd.to_datetime(x) if isinstance(x, np.datetime64) else x
    for i in ['x0', 'x1']:
        data[i] = format_timestamp(data[i])

    shp_specs = get_shape_specs(sh_specs, sh_type, xref, yref)

    for d in data:
        shp_specs.update({d: data[d]})

    shp = list(fig.select_shapes()) + [shp_specs]
    fig.update_layout(shapes=shp)


def draw_shape(fig, drag_mode='rect', sh_specs=None, add_buttons=True):
    """
    Manually draw shapes on plotly figure

    :param fig: plotly figure object
    :param drag_mode: str. One of 'closedpath','openpath', 'line',
    'rect' (Default), 'circle'
    :param sh_specs: dictionary. specs of shape to be plotted. Default: None
    (use defaults)
    :param add_buttons: bool. Whether to add buttons to draw shapes on
    plotly's figure control panel. Default: True
    :return:
    """
    if drag_mode is not None:
        shp_specs = get_shape_specs(sh_specs, drag_mode, None, None)
        for k in ['type', 'xref', 'yref']:
            shp_specs.pop(k)
        fig.update_layout(dragmode=f'draw{drag_mode}', newshape=shp_specs)
    if add_buttons:
        config = {'modeBarButtonsToAdd': ['drawline',
                                          'drawopenpath',
                                          'drawclosedpath',
                                          'drawcircle',
                                          'drawrect',
                                          'eraseshape'
                                          ]}
    else:
        config = {}
    fig.show(config=config)
