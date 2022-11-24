# ©, 2019, Sirris
# owner: HCAB

"""Helper functions to visualize distributions"""

from starterkits.preprocessing import normalization
from starterkits.visualization import vis_tools
import seaborn as sns
import numpy as np
import logging
import re
from matplotlib import pyplot as plt

logging.getLogger('matplotlib').setLevel(logging.ERROR)


def sized_boxplot(df, normalize=False, group_col=None,
                  normalize_method='z_score', width=12,
                  height_per_col=0.4):
    """
    Produce a boxplot for each numeric column.

    :param df : DataFrame.
    :param normalize: boolean. If True, data will be normalized.
    :param group_col: string.  Column to group by when normalizing (optional).
    :param normalize_method: See tools.preprocessing.normalization.normalize
    :param width: int.  Width of plot in inches (default 12)
    :param height_per_col: Height of plot per column in inches (default 0.4).
    :returns Seaborn plot.
    """
    if normalize:
        df = normalization.normalize(
            df,
            cols_to_normalize=list(set(df.columns) - {group_col}),
            group_col=group_col, method=normalize_method)

    # drop non-numeric columns
    df = df.select_dtypes(['number'])
    plt.figure(figsize=(width, height_per_col * len(df.columns)))
    g = sns.boxenplot(data=df, orient='h')
    return g


def histograms(df, bins=30, sample=False, fig_size=None, fig_layout=None,
               multi_color=False, data_type='numeric', plot_dist=True):
    """
    Produce a histogram for numeric or non-numeric columns, in a facet grid.

    :param df : DataFrame.
    :param bins: int.  Number of bins per histogram.
    :param sample: int.  Number of rows to sample from the dataframe.  If the
    data frame has fewer rows than this parameters, the entire dataframe is
    plotted. Default: None (no sampling)
    :param fig_size : tuple. Size of matplotlib plot. Default : None (will
     be determined)
    :param fig_layout : tuple. (nrows, ncols). Default : None (will be
    determined)
    :param multi_color : bool. Whether to make each plot with a different
    color. Default : False
    :param data_type : str. What type of data to plot. If numeric, provide any
    string matching regex '^num|^int|^float' (ex: 'number','numeric',
    'integer',...). Otherwise, will plot non-numeric columns. Default :
    'numeric'
    :param plot_dist: bool. Whether to plot density on top of histogram.
    Default: True
    :returns histogram plot.
    """
    # subsample, if requested
    if sample:
        df = df.sample(sample if sample < len(df) else len(df))

    # subset dataset
    if re.match('^num|^int|^float', data_type):
        data_types = ['number']
        data_type = 'numeric'
    else:
        data_types = ['object', 'category']
        data_type = 'string'
    d0 = df.select_dtypes(data_types).replace([np.inf, -np.inf], np.nan)

    # define figure layout and colors
    fig_layout = fig_layout or vis_tools.optimal_layout(d0.shape[1])
    fig, axes = plt.subplots(fig_layout[0], fig_layout[1],
                             figsize=fig_size or (fig_layout[1] * 5,
                                                  fig_layout[0] * 5))
    axes = axes.flatten()
    if multi_color:
        cols = vis_tools.define_xkcd_categories(d0.columns)
    else:
        cols = {d: sns.xkcd_rgb['cool blue'] for d in d0.columns}

    # loop through columns and visualize
    ct = -1
    for k, v in enumerate(d0.columns):
        if data_type == 'string':
            sns.countplot(d0[v], ax=axes[k], color=cols[v])
            vis_tools.format_axes(axes[k], rotate=45)
        else:
            sns.histplot(d0[v].dropna(), color=cols[v], bins=bins,
                         kde=plot_dist, ax=axes[k])
            # axes[k].hist(d0[v].dropna(), color=cols[v], bins=bins)

        axes[k].set_xlabel('')
        axes[k].set_title(v)
        axes[k].legend().remove()
        k_loc = [k * fig_layout[1] for k in np.arange(fig_layout[0])]
        if k in k_loc:
            axes[k].set_ylabel('Count')
        else:
            axes[k].set_ylabel('')
        ct += 1
    plt.tight_layout()
    vis_tools.clean_remaining(axes, ct)

    axes = axes.reshape(fig_layout)
    return axes
