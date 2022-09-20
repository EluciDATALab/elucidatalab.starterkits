# ©, 2019, Sirris
# owner: HCAB

"""support functions for matplotlib plotting"""
import numpy as np
import seaborn as sb
import os
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize, \
    ListedColormap
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
import re


def define_color_categories(categories, cmap=None):
    """
    Create dictionary with a color from a color palette for each category.

    Keys of the dictionary will correspond to the categories and the dictionary
    can be passed to most visualizations in seaborn (at least) using the
    palette=dict option. A list of all palettes can be found here:
    https://matplotlib.org/users/colormaps.html
    :param categories : list. List of categories.
    :param cmap : str. Name of colorpalette to be passed to seaborn
    color_palette. It can also be a list of palettes, in case the number of
    categories exceed the number colors. Default : None (will pick from
    seaborn palettes).
    :returns: Dictionary with categories as keys and color as values
    """
    palettes = ['Dark2', 'Set1', 'Set2', 'Set3', 'Accent']
    if cmap is None:
        cmap = palettes
    elif not isinstance(cmap, list):
        cmap = [cmap]

    col_maps = np.vstack([np.array(sb.color_palette(c)) for c in cmap])
    cols = {c: col_maps[k, :] for k, c in enumerate(categories)}

    return cols


def define_xkcd_categories(categories):
    """
    Create dictionary with a color from an xkcd color.

    Keys of the dictionary will correspond to the categories and the dictionary
    can be passed to most visualizations in seaborn (at least) using the
    palette=dict option. A list of all palettes can be found here:
    https://xkcd.com/color/rgb/
    :param categories : list. List of categories.
    :returns: Dictionary with categories as keys and color as values
    """
    cols_xkcd = np.random.choice(list(sb.xkcd_rgb.values()),
                                 len(categories),
                                 replace=False)

    cols = {c: cols_xkcd[k] for k, c in enumerate(categories)}

    return cols


def define_color_palette(col_dict):
    """
    Make a custom color palette from a dictionary containing category names

    as keys and colors as values
    :param col_dict: dict. Dictionary with categories as keys and color as
    values
    :returns: matplotlib colormap
    """
    cm = ListedColormap(col_dict.values())
    return cm


def define_linear_colormap(seq):
    """
    Return a LinearSegmentedColormap

    From [https://stackoverflow.com/questions/16834861/create-own-colormap-
    using-matplotlib-and-plot-color-scale]
    usage: cm = make_colormap([(0.1, 0.3, 0.2), (0.5, 0.4, 0.5), (0, 0, 0)])
    cb = plt.imshow(X, cmap=cm)
    plt.colorbar(cb)
    :param seq: a sequence of floats and RGB-tuples. The floats should be
    increasing and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def define_color_patches(col_dict):
    """
    Add a custom made legend to a matplotlib axes using the colors in the

    provided dictionary.
    Can be fed to plt.legend using input handles.
    Ex: axes.legend(handles=define_color_patches(col_dict))

    :param col_dict: dict. Dictionary with categories as keys and color as
    values
    :returns: color legend for matplotlib axes
    """
    return [mpatches.Patch(color=col_dict[a], label=a) for a in col_dict]


def format_axes(axes=None, is_date=True, date_format='%Y-%m-%d %H:%M:%S',
                rotate=45, which_axis='x', font_size=None):
    """
    Format x-axis labels

    :param axes: matplotlib axes object. Default: none (get with plt.gca())
    :param is_date : bool. Whether axis is to be formatted as date object.
    Default: True
    :param date_format : str. Datetime formatting string. Default : None.
    (don't format)
    :param rotate : int. Whether to rotate the strings. Default : 45
    (use none for no rotation)
    :param which_axis : str. Which axis to format. Options are ['x', 'y'].
    Default : 'x'
    :param font_size: int. Font size to use.
    Default: None (use function default)
    """
    axes = axes or plt.gca()
    if is_date:
        a = axes.xaxis if which_axis == 'x' else axes.yaxis
        a.set_major_formatter(mdates.DateFormatter(date_format))

    if rotate is not None:
        for tick in eval('axes.get_%sticklabels()' % which_axis):
            tick.set_rotation(rotate)
            tick.set_horizontalalignment('right')
    if font_size is not None:
        for tick in eval('axes.get_%sticklabels()' % which_axis):
            tick.set_fontsize(font_size)

    return axes


def axes_font_size(xlabel, ylabel, title, axes=None, legend=None,
                   tick_size=14, label_size=16, title_size=18,
                   legend_kwargs=None, format_axes_kwargs=None):
    """
    Format size of axes fonts

    :param xlabel : str. Label of x-axis
    :param ylabel : str. Label of y-axis
    :param title : str. Title
    :param axes: matplotlib axes object. Default: none (get with plt.gca())
    :param legend : str. Legend name. Default: None
    :param tick_size : int. Fontsize for x/y ticks. Default : 14
    :param label_size : int. Fontsize for x/y ticks. Default : 16
    :param title_size : int. Fontsize for x/y ticks. Default : 18
    :param legend_kwargs: dict. Further arguments to be passed to legend.
    Default: None ({})
    :param format_axes_kwargs: dict. Further arguments to be passed to
    format_axes. Default: None ({})
    """
    axes = axes or plt.gca()
    legend_kwargs = {} if legend_kwargs is None else legend_kwargs
    format_axes_kwargs = ({} if format_axes_kwargs is None
                          else format_axes_kwargs)
    format_axes_kwargs['is_date'] = format_axes_kwargs.get('is_date', False)
    format_axes_kwargs['rotate'] = format_axes_kwargs.get('rotate', None)

    axes.set_xlabel(xlabel, fontsize=label_size)
    axes.set_ylabel(ylabel, fontsize=label_size)
    for i in ['x', 'y']:
        format_axes(axes,
                    which_axis=i,
                    font_size=tick_size,
                    **format_axes_kwargs)
    axes.set_title(title, fontsize=title_size)

    if legend is not None:
        legend_kwargs['fontsize'] = legend_kwargs.get('legend_size', 16)
        legend_kwargs['markerscale'] = legend_kwargs.get('legend_marker_size',
                                                         2)

        legend_marker_alpha = legend_kwargs.get('legend_marker_alpha', 1)
        legend_title_size = legend_kwargs.get('legend_title_size', 18)
        legend_kwargs = {c: legend_kwargs[c] for c in legend_kwargs
                         if c not in ['legend_marker_alpha',
                                      'legend_title_size',
                                      'legend_size',
                                      'legend_marker_size']}
        lgd = axes.legend(title=legend, **legend_kwargs)
        plt.setp(lgd.get_title(), fontsize=legend_title_size)
        plt.setp(lgd.get_lines(), alpha=legend_marker_alpha)
        for lg in lgd.legendHandles:
            lg.set_alpha(legend_marker_alpha)


def change_width(new_value, ax=None):
    """
    Change width of bars in seaborn plot.

    Adapted from
    [https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-
    bar-chart-created-using-seaborn-factorplot]

    :param ax: matplotlib axes object. Default: none (get with plt.gca())
    :param new_value: int. bar width
    """
    ax = ax or plt.gca()
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def optimal_layout(n_facets, layout_format='2:3'):
    """
    Define optimal number of rows and columns according to the number of facets

    to plot and the desired layout format

    :param n_facets : int. Number of facets to plot
    :param layout_format : str. Ratio to use for the faceting format in the
    format
    'height:width'. Default : '2:3'

    :returns: tuple with number of rows and columns
    """
    ratio = [int(s) for s in layout_format.split(':')]

    n_rows = np.ceil(np.sqrt(n_facets * ratio[0] / ratio[1]))

    n_cols = np.ceil(n_facets / n_rows)

    return int(n_rows), int(n_cols)


def clean_remaining(axes, id_last):
    """
    Make non-used axes empty

    :param axes: matplotlib axes object.
    :param id_last : int. Index of last plotted facet
    """
    for i in np.arange(id_last + 1, len(axes)):
        axes[i].axis('off')


def set_twin_yaxis(axes=None, label=None, ax_color=None):
    """
    Setup a secondary y-axis on the right side

    :param axes: matplotlib axes object. Default: none (get with plt.gca())
    :param label : str. Name of secondary y-axis label. Default : None
    :param ax_color : str. Color to use for twin y-axis. Default : None
    :returns: axes object
    """
    axes = axes or plt.gca()
    ax2 = axes.twinx()
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    if label is not None:
        ax2.set_ylabel(label)
    if ax_color is not None:
        ax2.yaxis.label.set_color(ax_color)

    return ax2


def add_inset_axes(x, y, w, h, ax=None):
    """
    Add inset axes to axes with specified width, height and position.

    Wrapper around [https://matplotlib.org/api/_as_gen/mpl_toolkits.axes_
    grid1.inset_locator.inset_axes.html]

    :param x: float. Relative position of the inset on the x-axis
    :param y: float. Relative position of the inset on the y-axis
    :param w : str or float. Width of axis to create. See reference
    :param h : str or float. Height of axis to create. See reference
    :param ax: matplotlib axes object. Default: none (get with plt.gca())
    :returns: axes object
    """
    ax = ax or plt.gca()
    return ax.inset_axes([x, y, w, h])


def add_colorbar(v_max, ax=None, v_min=0, cmap='Blues', title='',
                 orientation='vertical'):
    """
    Add custom colorbar to axes

    :param v_max : float. Maximum value of colorbar scale
    :param ax: matplotlib axes object. Default: none (get with plt.gca())
    :param cmap : str. Name of colormap. Default : 'Blues'
    :param v_min : float. Minimum value of colorbar scale. Default = 0
    :param title : str. Title of colorbar. Default = ''
    :param orientation : str. Orientation of colorbar. Default = 'vertical'
    """
    ax = ax or plt.gca()
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb1 = plt.colorbar(sm, ax=ax, orientation=orientation)
    cb1.set_label(title)


class MidpointNormalize(Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side

    from a prescribed midpoint value)

    e.g. im=ax1.imshow(array,
                       norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    from http://chris35wills.github.io/matplotlib_diverging_colorbar/
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """Define parameters"""
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Call"""
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def savefig_to_dir(data_dir, output_type='png', dpi=300,
                   bbox_inches='tight', transparent=True,
                   kwargs_savefig=None):
    """
    Returns a function to save figures to a given data directory.

    It's useful when several figures in one notebook will be saved to the same
    location.
    Example usage:
        savefig = vis_tools.savefig_to_dir('/Users/hcab/Documents/media')
    plt.plot(x, y)
    savefig('myfigure')

    This will save the plot to a figure called 'myfigure.png' in
    '/Users/hcab/Documents/media'
    The function call accepts an optional input 'transparency', a boolean to
    define the transparency of the figure. Default = True

    :param data_dir : str. Location where to save figures
    :param output_type: str. File type of figures. default: png
    :param dpi : int. Image resolution. Default : 300
    :param bbox_inches : str. Only the given portion of the figure is saved.
    If 'tight', try to figure out the tight bbox of. Default: 'tight'.
    :param transparent: bool. Whether to have a transparent background.
    Default: True
    :param kwargs_savefig: dict. Further arguments to plt.savefig.
    Default: None
    See matplotlib.pyplot.savefig for more details.
    :returns: function
    """
    kwargs_savefig = kwargs_savefig or {}
    if os.path.isdir(data_dir) is False:
        os.mkdir(data_dir)

    def savefig(fname):
        plt.savefig(os.path.join(data_dir, '%s.%s' % (fname, output_type)),
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    transparent=transparent,
                    **kwargs_savefig)

    return savefig


def add_facet_axes_labels(axes,
                          nrow,
                          ncol,
                          xlabel,
                          ylabel,
                          titles=None,
                          kwargs_axes_font=None):
    """
    This function adds xlabels to the bottom row of axes and ylabels to the

    left most column in a faceted figure.

    :param axes: array of axes
    :param nrow: int. Number of rows in figure
    :param ncol: int. Number of cols in figure
    :param xlabel: str. Label for x-axis
    :param ylabel: str. Label for y-axis
    :param titles: list/array. Titles for each subfigure. If None will be set
    to empty (Default)
    :param kwargs_axes_font: dict. Further arguments to axes_font_size
    :return:
    """
    kwargs_axes_font = {} if kwargs_axes_font is None else kwargs_axes_font
    if len(axes.shape) == 2:
        axes = axes.flatten()
    if titles is None:
        titles = np.repeat('', len(axes))
    if len(titles) != len(axes):
        raise ValueError('titles input should be the same number as axes in '
                         'the figure')

    for k, a in enumerate(axes):
        xlab = xlabel if k >= (nrow * ncol - ncol) else ''
        ylab = ylabel if k in np.arange(0, nrow * ncol, ncol) else ''
        axes_font_size(xlab, ylab, titles[k], axes=a, **kwargs_axes_font)


def get_linear_cmap(colors):
    """
    Create a linear colormap from two or more colors.

    :param colors: list. List of colors that are used to create the linear
    colormap
    :return: colormap
    """
    cm = LinearSegmentedColormap.from_list('my_cmap', colors)

    return cm


def prettify(x):
    """
    Format string to make it look prettier (replace underscores by spaces and

    capitalize first letter)

    :param x: str or list
    :return: formatted string
    """
    def _prettify(s):
        s_split = re.split('[ _]', s)
        s_split[0] = s_split[0][0].upper() + s_split[0][1:]
        return ' '.join(s_split)

    x = [x] if isinstance(x, str) else x
    x_out = [_prettify(s) for s in x]
    x_out = x_out[0] if len(x_out) == 1 else x_out

    return x_out


def remove_spines(ax=None, sides='lrtb'):
    """
    Remove spines from plot

    :param ax: matplotlib axes object. Default: none (get with plt.gca())
    :param sides: str. String specifying sSides to remove spines from: l(left),
    r(right), t(top), b(bottom). Default: lrtb
    """
    ax = ax or plt.gca()
    sides_lib = {'l': 'left', 'r': 'right', 't': 'top', 'b': 'bottom'}
    for s in sides:
        ax.spines[sides_lib[s]].set_visible(False)


def color_spines(color, ax=None, sides='lrtb', linewidth=1):
    """
    Color spines from plot

    :param color: str or color object. Color for spines
    :param ax: matplotlib axes object. Default: none (get with plt.gca())
    :param sides: str. String specifying sSides to remove spines from: l(left),
    r(right), t(top), b(bottom). Default: lrtb
    :param linewidth: int. Linewidth of spines. Default: 1
    """
    ax = ax or plt.gca()
    sides_lib = {'l': 'left', 'r': 'right', 't': 'top', 'b': 'bottom'}
    for s in sides:
        ax.spines[sides_lib[s]].set_color(color)
        ax.spines[sides_lib[s]].set_linewidth(linewidth)
