# ©, 2022, Sirris
# owner: MDHN

"""Function to plot circular heatmaps."""
import math
from matplotlib import colors
import matplotlib.pyplot as plt


def plot_circular_heatmap(table, cmap='Blues', vmin=None, vmax=None, vcenter=None, inner_r=0.25, line_width=0.5,
                          colorbar_name='', title='', hide_labels=False, radial_label_step=1, radius_label_step=1,
                          radial_label_size=None, radius_label_size=None, inner_labels_color='white', pie_args=None,
                          colorbar=True, colorbar_scale_factor=0.7, ax=None):
    """
    Plot a circular heatmap, based on overlapping pie charts.

    source: https://medium.com/analytics-vidhya/exploratory-data-analysis-of-google-fit-data-with-pandas-and-seaborn-
            a4369366c543

    :param table:       DataFrame   Each row represents a circle, with the first row being the outer circle
    :param cmap:        Object      Matplotlib colormap. E.g. 'Blues', 'Purples', 'rainbow', 'turbo', ...
                                    See: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :param vmin:        Integer     Value mapped to the first color of the continuous color scale
    :param vmax:        Integer     Value mapped to the last color of the continuous color scale
    :param vcenter:     Integer     Value mapped to the center color of the continuous color scale
    :param param inner_r: Float     Radius of the circular white center of the circular heatmap. Range: ]0:inf]
    :param line_width:  Float       Width of the circular and radial grid lines
    :param colorbar_name: String    Title of the legend
    :param title:       String      Title of the heatmap
    :param hide_labels: Bool        Whether or not to show axis labels
    :param radial_label_step:   Int     Step to show bin labels along the angle dimension (useful if many radial bins).
                                        This value should be minimum 1
    :param radius_label_step:   Int     Step to show bin labels along the distance from the center dimension
                                        (useful if many circles). This value should be minimum 1
    :param radial_label_size:   Float   Font size of radial labels
    :param radius_label_size:   Float   Font size of radius labels
    :param inner_labels_color:  String  Color of the inner labels
    :param param pie_args:      Dict.   Additional args for the pie plots
    :param colorbar:            Bool    Whether or not to show the colorbar/legend
    :param colorbar_scale_factor:   Float       Fraction by which to multiply the size of the colorbar.
                                                Range: ]-inf:+inf[
    :param ax:                      plt.axis    Axis which will be used to plot the heatmap

    ------------------------------------------------------------------------------------------

    Example:
    -------
        import pandas as pd
        import matplotlib.pyplot as plt
        from starterkits.visualization import circular_heatmap as ch

        week_1 = [[0.8, 0.6, 0.5, 0.4, 0.7, 0.2, 0.3],
                  [1.1, 1.2, 2.1, 2.2, 1.3, 1.1, 1.2],
                  [2.4, 2.2, 2.5, 2.3, 2.1, 2.2, 2.2],
                  [2.8, 2.2, 2.3, 2.7, 2.2, 2.3, 2.4],
                  [2, 2.6, 2.5, 2.4, 2.7, 2.6, 2.5]]
        week_2 = [[0.6, 0.4, 0.3, 0.3, 0.4, 0.1, 0.3],
                  [0.7, 0.8, 1, 1, 0.9, 0.8, 0.8],
                  [1.1, 0.9, 1, 1.1, 1, 1, 1],
                  [1.1, 1, 0.9, 1, 0.9, 1.1, 0.9],
                  [0.9, 1.1, 1.1, 0.8, 0.7, 1, 0.9]]
        indices = ["Night", "Evening", "Afternoon", "Noon", "Morning"]
        columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        df_week1 = pd.DataFrame(data=week_1, index=indices, columns=columns)
        df_week2 = pd.DataFrame(data=week_2, index=indices, columns=columns)

        # Plot one single heatmap:
        ax = ch.plot_circular_heatmap(df_week1, inner_r=0.1, colorbar_name='Dummy data', vmin=0, vcenter=1,
                                      radial_label_size=14, radius_label_size=14, title='Example circular heatmap',
                                      cmap='Purples')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0] = ch.plot_circular_heatmap(df_week1, inner_r=0.5, vmin=0, vmax=2.8,
                                           title='Week 1', colorbar=False, ax=axes[0])
        axes[1] = ch.plot_circular_heatmap(df_week2, inner_r=0.1, colorbar=False, vmin=0, vmax=2.8,
                                           title='Week 2', inner_labels_color='black', line_width=5, ax=axes[1])
        axes[2] = ch.plot_circular_heatmap(df_week2, inner_r=0.001, colorbar_name='Dummy data', vmin=0, vmax=2.8,
                                           title='Week 2', hide_labels=True, line_width=0, ax=axes[2])

    """
    if pie_args is None:
        pie_args = {}

    vmin = table.min().min() if vmin is None else vmin
    vmax = table.max().max() * 1.1 if vmax is None else vmax

    # if vcenter is defined, it might overwrite vmax or vmin
    vmin = min(vcenter - (vmax - vmin) * 0.1, vmin) if vcenter is not None else vmin
    vmax = max(vcenter + (vmax - vmin) * 0.1, vmax) if vcenter is not None else vmax

    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.axes()

    ax = _create_white_centre_circle(inner_r, ax)

    cmapper = _get_color_mapper(vmin, vmax, vcenter, cmap)

    ax = _construct_heatmap(table, cmapper, hide_labels, radius_label_step, radial_label_step,
                            radial_label_size, radius_label_size, inner_labels_color, pie_args,
                            inner_r, line_width, ax)

    if colorbar:
        cb = plt.colorbar(cmapper, label=colorbar_name, shrink=colorbar_scale_factor, ax=ax)
        cb.outline.set_visible(False)  # remove colorbar frame

    ax.set_title(title, pad=inner_r * 100)

    return ax


def _create_white_centre_circle(inner_r, ax):
    centre_circle = plt.Circle((0, 0), inner_r, edgecolor='white', facecolor='white', fill=True, linewidth=0.25)
    ax.add_artist(centre_circle)
    return ax


def _get_color_mapper(vmin, vmax, vcenter, cmap):
    if vcenter is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def _construct_heatmap(table, cmapper, hide_labels, radius_label_step, radial_label_step,
                       radial_label_size, radius_label_size, inner_labels_color, pie_args, inner_r, line_width, ax):
    """This is done by overlapping pie charts."""
    n, m = table.shape

    for i, (row_name, row) in enumerate(table.iterrows()):
        # Mark nan points in grey
        mapped_colors = [cmapper.to_rgba(x) if not math.isnan(x) else (0.8, 0.8, 0.8) for x in row.values]
        labels = None if i > 0 or hide_labels else [label if i % radial_label_step == 0 else ""
                                                    for i, label in enumerate(table.columns)]
        wedges = ax.pie([1] * m, radius=inner_r + float(n - i) / n, colors=mapped_colors,
                        labels=labels, startangle=90, counterclock=False, wedgeprops={'linewidth': -1},
                        textprops={'fontsize': radial_label_size}, **pie_args)
        plt.setp(wedges[0], edgecolor='white', linewidth=line_width)

        if not hide_labels:
            wedges = _set_labels(wedges, i, radius_label_step, ax, inner_r,
                                 n, row_name, radius_label_size, inner_labels_color, line_width)

        plt.setp(wedges[0], edgecolor='white', linewidth=line_width)

    return ax


def _set_labels(wedges, i, radius_label_step, ax, inner_r, n, row_name, label_size, inner_labels_color, line_width=5):
    """This function makes sure the labels are correctly added to the plot."""
    # Center the radial labels:
    for label in wedges[1]:
        label.set_horizontalalignment('center')

    # Add the circular labels:
    if i % radius_label_step == 0:
        label_distance = (n - i) / n + inner_r - (0.5 / n)

        wedges = ax.pie([1], radius=inner_r + float(n - i - 1) / n, colors=['w'], labels=[row_name],
                        startangle=-90, wedgeprops={'linewidth': 0}, labeldistance=1.05,
                        textprops=dict(fontsize=label_size, position=(0.01 + line_width/100, label_distance),
                                       color=inner_labels_color, weight='extra bold', family='monospace'))

    return wedges
