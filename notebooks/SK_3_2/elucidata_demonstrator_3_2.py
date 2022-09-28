# coding: utf-8
"""Imports, classes and functions for the SK 3.2 notebook."""
import warnings

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from matplotlib import rcParams

# global configuration options
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = (16, 4)

REQUIRED_PACKAGE_VERIONS = [
    "datetime==4.3",
    "matplotlib==2.2.3",
    "pandas==0.23.4",
    "numpy==1.15.4",
    "seaborn==0.9.0",
]


def try_package(x):
    pkg, pkg_version = x.split('==')
    try:
        version = pkg_resources.get_distribution(pkg).version
        if pkg_version != version:
            print(f'Warning: This notebook was tested with '
                  f'version {pkg_version} of {pkg!r},'
                  f'but you have version {version} installed.')
    except Exception as e:
        print(e)


def assert_correct_package_versions():
    for p in REQUIRED_PACKAGE_VERIONS:
        try_package(p)


def plot_boxplot(df, x, y, y_scale='linear'):
    """Draw blox plot.

    Args:
        df (DataFrame): data to plot
        x (str): column of `df` to use as x-axis
        y (str): column of `df` to use as y-axis
        y_scale (str): 'linear' or 'log'

    Return: (matplotlib Axes): the Axes object with the plot drawn onto it
    """
    field_unique = df[x].unique()

    fig, ax = plt.subplots(figsize=(8, 8))
    cols = sns.color_palette('Dark2')
    cols = {e: cols[k] for k, e in enumerate(field_unique)}
    sns.boxplot(x=x, y=y, data=df, whis=np.inf, hue=x,
                palette=cols, notch=True, boxprops=dict(alpha=.3), ax=ax,
                dodge=False)
    sns.stripplot(x=x, y=y, data=df, hue=x,
                  palette=cols, jitter=True, size=10, alpha=0.7, ax=ax)
    ax.get_legend().remove()
    ax.set_title(f'Relationship Between the {x} and the {y} variables')
    for a in cols:
        ax.axhline(df[df[x] == a][y].median(), color=cols[a], linestyle='--')
    ax.set_yscale(y_scale)

    return ax


def plot_barplot_with_labels(df, x, y, labels, title, ymax=1,ax=None):
    """Draw bar plot with labels centered on top of the bars.

    Args:
        df (DataFrame): data to plot
        x (str): column of `df` to use as x-axis
        y (str): column of `df` to use as y-axis
        labels (str): column of `df` to use as bar labels
        title (str): title of bar plot
        ymax (int or float): y-axis max
        ax (matplotlib Axes): the Axes object with the plot drawn onto it

    Return: (matplotlib Axes): the Axes object with the plot drawn onto it
    """
    rcParams['figure.figsize'] = (12, 6)

    if ax==None:
        g = sns.barplot(x=x, y=y, data=df)
    else:
        g = sns.barplot(x=x, y=y, data=df, ax=ax)

    g.set_title(title)
    g.set_ylim([0, ymax])
    g.grid("on", axis="y")
    for patch, label in zip(g.patches, df[labels]):
        x = patch.get_x() + patch.get_width()/2  # center label on bar
        y = 0 #patch.get_height()
        g.annotate(label, (x, y), ha='center', va='bottom', color='black')
    g.set_xticklabels(g.get_xticklabels(), ha="right", rotation =30)

    return g

def plot_double_barplot(df,ax,title,x ='to_station_name',y='ratio_f2m',secondary_y='ratio_m2f',ymax=9):
    """Draw bar plot with a secundary axis.

    Args:
        df (DataFrame): data to plot
        ax (matplotlib Axes): the Axes object with the plot drawn onto it
        title (str): title of bar plot
        x (str): column of `df` to use as x-axis
        y (str): column of `df` to use as y-axis
        secondary_y (str): column of `df` to use as secundary y-axis
        ymax (int): y-axis max.

    Return: (matplotlib Axes): the Axes object with the plot drawn onto it
    """
    df.plot(ax=ax, kind= 'bar' , x=x, y=[y,secondary_y]  ,rot=30)
    ax.set_ylabel('ratio')
    ax.set_xticklabels(ax.get_xticklabels(),ha="right")
    ax.yaxis.set_ticks_position('left')
    ax.axhline(2,c='k',ls='--')

    ax.set_ylim(bottom=0, top=ymax)
    ax.set_title(title)

    return ax


class Map:
    """Wraps a folium map.

    Allows to add markers stored in a DataFrame, see `add_markers`.
    Supports display in Jupyter notebook.
    """
    def __init__(self):
        self.map = folium.Map(
            location=[47.642394, -122.323738],
            tiles='openstreetmap',
            control_scale=True,
            zoom_start=12,
            min_zoom=12,
            max_zoom=18,
        )

    def _repr_html_(self):
        """Display the HTML map in a Jupyter notebook."""
        return self.map._repr_html_()

    def add_markers(self, df, lat='lat', lon='lon', radius=None, scale=0.08,
                    marker=folium.CircleMarker, popup_name='name', color='blue'):
        """Add markers to map.

        Args:
             df (DataFrame): marker positions
             lat (str): field in `df` with latitude values
             lon (str): field in `df` with longitute values
             radius (str or None): field in `df` with radii
                if None, use 100
             marker (folium marker)
                e.g., folium.Circle  # radius in map units
                e.g., folium.CircleMarker  # radius in screen units
             popup_name (str): field in `df` with marker popup names
             color (str): color of markers (one color for all markers)

        Return: self
        """
        if radius is None:
            radii = 100 * np.ones(len(df))  # default
        elif radius == 'elevation':
            radii = 2 * df[radius].values
        else:
            radii = scale * df[radius].values

        radius_column_name_to_label = {
            "count": "Count",
            "AbsTripDifference": "Unbalance",
            "elevation": "Elevation (m)"
        }
        radius_label = radius_column_name_to_label[radius] if radius in radius_column_name_to_label else ""

        for i in range(0, len(df)):  # todo: iterrows?
            marker(
                location=[df.iloc[i][lat], df.iloc[i][lon]],
                radius=radii[i],
                popup=df.iloc[i][popup_name] + (f" - {radius_label}: {np.around(df.iloc[i][radius],1)}" if radius is not None else ""),
                color=color,
                fill_color=color,  # use same color for the fill
            ).add_to(self.map)

        return self  # so we can chain member calls


def extract_destination_station_frequency_per_gender(df_trips):
    """Extract per gender (female and male) how frequently each destination
    station was reached for trips.

    Args:
        df_trips (DataFrame): bike trips

    Return: (DataFrame) frequency and ratios per gender per station
        The ratios ('ratio_f2m' and 'ratio_m2f') are the ratio of the frequency
        between genders. This ratio informs to what extent the station is more
        popular for one gender w.r.t. the other one.
    """
    nof_trips_women = len(df_trips[df_trips['gender'] == 'Female'])
    nof_trips_men = len(df_trips[df_trips['gender'] == 'Male'])

    stations_women = (df_trips[df_trips['gender'] == 'Female']
        .groupby(['gender', 'to_station_name'])
        .agg({'bikeid': 'count',
              'to_station_lat': 'first',
              'to_station_lon': 'first'})
        .rename(columns={'bikeid': 'frequency'}))
    stations_women['frequency'] = stations_women['frequency'] / nof_trips_women

    stations_men = (df_trips[df_trips['gender'] == 'Male']
        .groupby(['gender', 'to_station_name'])
        .agg({'bikeid': 'count'})
        .rename(columns={'bikeid': 'frequency'}))
    stations_men['frequency'] = stations_men['frequency'] / nof_trips_men

    ans = pd.merge(stations_women, stations_men, on=['to_station_name'],
                   suffixes=('_female', '_male'))
    ans['ratio_f2m'] = ans['frequency_female'] / ans['frequency_male']
    ans['ratio_m2f'] = ans['frequency_male'] / ans['frequency_female']

    return ans


