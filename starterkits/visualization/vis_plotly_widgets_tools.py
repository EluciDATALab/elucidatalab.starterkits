# ©, 2019, Sirris
# owner: HCAB

"""support functions for vis_plotly_widgets"""
import pandas as pd
import re
import inspect
from copy import deepcopy
from plotly.graph_objects import Layout
import ipywidgets as widgets


def is_in_func_args(x, fun):
    """
    Test if name is part of function argument list

    :param x: str. name of argument
    :param fun: function. Function to test
    :return:
    bool
    """
    fun_args = inspect.signature(fun)
    return x in fun_args.parameters


def get_controller(controller):
    """
    Parse dictionary into widget.

    :param controller: dict. Dictionary with inputs to widget. Must have at
    least key indicating name of widget
    :return:
    widget
    """
    controller = deepcopy(controller)
    wdgt = controller['widget']
    if wdgt == 'SelectionRangeSlider':
        if (('start_date' not in controller.keys())
                | ('end_date' not in controller.keys())):
            raise Exception(
                'If selecting a SelectionRangeSlider widget you must provide '
                'with start_date and end_date values')
        controller_out = define_slider_date_range(
            **{c: controller[c] for c in controller
               if is_in_func_args(c, define_slider_date_range)})
    else:
        controller.pop('widget')
        controller['continuous_update'] = controller.get('continuous_update',
                                                         False)
        controller['style'] = controller.get('style',
                                             {'description_width': 'initial'})
        controller_out = getattr(widgets, wdgt)(**controller)
    return controller_out


def define_slider_date_range(start_date,
                             end_date,
                             freq='D',
                             date_format='%d/%m/%Y'):
    """
    Define slider date range widget

    :param start_date: str/datetime. starting date
    :param end_date: str/datetime. end date
    :param freq: str. Frequency of slider steps. Default: 'D'
    :param date_format: str. Date display format. Default: '%d/%m/%Y'
    :return:
    """
    dates = pd.date_range(start_date, end_date, freq=freq)

    options = [(date.strftime(date_format), date) for date in dates]
    index = (0, len(options) - 1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '400px'},
        continuous_update=False
    )

    return selection_range_slider


def expand_xlabel(kwargs, xlabel, list_args, only_last=True):
    """
    Propagate parameters to x-axis formatting to all axis in plot

    :param kwargs: dict. dictionary with parameters
    :param xlabel: str. name of xlabel
    :param list_args: list. List of possible arguments
    :param only_last: bool. Whether to keep only the last x-axis element.
    Default: True
    :return:
    """
    if xlabel is None:
        return kwargs
    else:
        key = [c for c in list_args if re.match('xaxis[0-9]*', c)]
        key = [sorted(key)[-1]] if only_last else key
        for c in key:
            ax = kwargs.get(c, {})
            ax.update({'title': xlabel})
            kwargs[c] = ax
    return kwargs


def update_layout(fig, ts_kwargs, layout_kwargs):
    """
    Update plotly figure layout with further parameters

    :param fig: plotly figure object
    :param ts_kwargs: dict. further arguments for figure layout from ts_kwargs
    (one of vis_plotly functions).
    :param layout_kwargs: dict. fig layout arguments to be updated
    :return:

    """
    kwargs = {**ts_kwargs, **layout_kwargs}
    list_args = list(Layout.__dict__.keys())
    kwargs = {c: kwargs[c] for c in kwargs if c in list_args}
    kwargs = expand_xlabel(kwargs, ts_kwargs.get('xlabel', None), list_args)
    fig.update_layout(kwargs)
