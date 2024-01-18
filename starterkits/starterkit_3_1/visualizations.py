# ©, 2022, Sirris
# owner: HCAB

import calmap
import folium
import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sb
from starterkits.visualization.circular_heatmap import plot_circular_heatmap
from starterkits.visualization import vis_plotly as vp
from starterkits.visualization import vis_plotly_widgets_tools as vpwt
from ipywidgets import interact
from pandas.tseries.holiday import USFederalHolidayCalendar
from pylab import rcParams
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

rcParams['figure.figsize'] = 15, 10

reordered_nodes = ['Fremont_Bridge', 'Spokane_St', '2nd_Ave', 'NW_58th_St', '39th_Ave', '26th_Ave']
reordered_columns = [f'Total {col}' for col in reordered_nodes]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def plot_map_nodes(data):
    """Plot the different nodes on a map"""
    NW58th_22nd = ['NW_58th_St', [47.670917, -122.384767]]
    NE39th_62nd = ['39th_Ave', [47.673974, -122.285785]]
    Spokane_street = ['Spokane_St', [47.571491, -122.345516]]
    SW26_oregon = ['26th_Ave', [47.562902, -122.365466]]
    N2nd_marion = ['2nd_Ave', [47.604466, -122.334486]]
    Fremont_bridge = ['Fremont_Bridge', [47.647707, -122.349715]]

    data.groupby('Node').agg({'Date': ['min', 'max'], 'Total': 'sum'})

    m = folium.Map(location=NW58th_22nd[1], zoom_start=11, min_zoom=11, max_zoom=18, scrollWheelZoom=False)

    for name, coordinates in [NW58th_22nd, NE39th_62nd, Spokane_street, SW26_oregon, N2nd_marion, Fremont_bridge]:
        circle_size = 200
        folium.Circle(location=coordinates, radius=circle_size, color='red',
                      fill_color='red', popup=name).add_to(m)

    return m


def get_totals_node(data):
    """Get total crossings for each node"""
    indexes = pd.Index(data['Timestamp'].unique(), name='Timestamp')
    data_totals_node = pd.DataFrame(index=indexes)
    for node in data['Node'].unique():
        data_node = data[data['Node'] == node]
        data_node = data_node.set_index('Timestamp')
        data_totals_node.loc[data_node.index, f'Total {node}'] = data_node['Total']
    data_totals_node = data_totals_node[reordered_columns]
    return data_totals_node


def normalize_and_resample(df, resample, normalize, columns):
    """Apply resampling and normalizing to the specified columns"""
    df = df.copy().reset_index().sample(10000)
    df = df.sort_values(by='Timestamp')

    if resample:
        df = df.resample(resample, on='Timestamp').sum().reset_index()
    if normalize:
        scaler = None
        if normalize == 'min-max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        if normalize == 'std':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    df = df.set_index('Timestamp')
    return df


def plot_activity_nodes(data):
    """Draw interactive plot of the number of crossings for each node"""
    df = get_totals_node(data)

    def make_plot(resample, normalize):
        df0 = normalize_and_resample(df, resample, normalize, reordered_columns)
        series = [df0[col] for col in reordered_columns]
        fig = vp.plot_multiple_time_series(series, show_time_frame=False, show_time_slider=False)
        fig.show(renderer="colab")

    controller_normalize = vpwt.get_controller({'widget': 'Dropdown',
                                                'options': [('No normalisation', False),
                                                            ('Min-Max scaling', 'min-max'),
                                                            ('Standard scaling', 'std')],
                                                'value': False,
                                                'description': 'Normalise:'})

    controller_resampling = vpwt.get_controller({'widget': 'Dropdown',
                                                 'options': [('No resampling', False),
                                                             ('Daily', 'D'),
                                                             ('Weekly', '7D'),
                                                             ('Monthly', 'M'),
                                                             ('Yearly', '365D')],
                                                 'value': False,
                                                 'description': 'Resampling rate'})

    interact(make_plot, resample=controller_resampling, normalize=controller_normalize)


def plot_calendar_heatmap(data):
    """Draw calendar heatmap of crossings for each year with interactive choice of color map"""
    def heatmap_with_cmap_selector(df, cmaps=None):
        cdict = {'green': ((0.0, 0.0, 0.0),   # no red at 0
                           (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                           (1.0, 0.8, 0.8)),  # set to 0.8 so it's not too bright at 1

                 'red': ((0.0, 0.8, 0.8),   # set to 0.8 so it's not too bright at 0
                         (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                         (1.0, 0.0, 0.0)),  # no green at 1

                 'blue': ((0.0, 0.0, 0.0),  # no blue at 0
                          (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                          (1.0, 0.0, 0.0))  # no blue at 1
                 }

        # Create the colormap using the dictionary
        RdGn = colors.LinearSegmentedColormap('RdGn', cdict)

        if cmaps is None:
            cmaps = {'Sequential (YlGn)': 'YlGn',
                     'Sequential (RdGn)': RdGn,
                     'Divergent (RdBu)': 'RdBu',
                     'Circular (hsv)': 'hsv'}
        default = 'YlGn'
        # define selector
        var_selector = widgets.Dropdown(options=cmaps, value=default,
                                        description='Color map', disabled=False)

        def _plot_heatmap(cmap):
            # noinspection PyTypeChecker
            data = df.groupby(['Date'])[['Total']].sum()
            fig, ax = calmap.calendarplot(data, monthticks=3,
                                          yearlabels=True, how=None, linewidth=0, cmap=cmap, fillcolor='lightgray',
                                          fig_kws=dict(figsize=(20, 10), edgecolor='gray'))
            fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist(), label='total crossings')
            plt.show()

        return _plot_heatmap, var_selector

    _plot_heatmap_function, var_selector_dropdown = heatmap_with_cmap_selector(data)
    interact(_plot_heatmap_function, cmap=var_selector_dropdown)


# noinspection PyUnusedLocal
def draw_heatmap(data, draw_colorbar=True, **kwargs):
    """Draw a heatmap of total crossings"""
    d = data.pivot(index='DayOfWeek', columns='Hour', values='Total')
    cbar_par = {'label': 'Total crossings', 'orientation': 'horizontal'} if draw_colorbar else None
    sb.heatmap(d, square=True, linewidths=0.1, xticklabels='', yticklabels='', cbar=draw_colorbar,
               cmap='YlGn', cbar_kws=cbar_par)


def draw_heatmap_crossings(data, node='Fremont_Bridge', draw_colorbar=True):
    """Draw a heatmap of total crossings for a specific node"""
    node_data = data[data.Node == node]
    node_df = node_data[['Total', 'DayOfWeek', 'Hour']].groupby(['DayOfWeek', 'Hour']).sum().reset_index()
    draw_heatmap(node_df, draw_colorbar)


def small_multiples_crossings(data):
    """Draw heatmaps of total crossings for each node in small multiples"""
    all_nodes_df = (data[['Total', 'DayOfWeek', 'Hour', 'Node']]
                    .groupby(['Node', 'DayOfWeek', 'Hour'])
                    .sum()
                    .reset_index())
    g = sb.FacetGrid(all_nodes_df, col='Node', col_wrap=3)
    g.map_dataframe(draw_heatmap, draw_colorbar=False)


def draw_circular_heatmap_crossings(data, node='Fremont_Bridge'):
    """Draw a circular heatmap of total crossings for a specific node"""
    node_data = data[data.Node == node]
    node_df = node_data[['Total', 'DayOfWeek', 'Hour']].groupby(['DayOfWeek', 'Hour']).sum().reset_index()
    node_df_pivoted = node_df.pivot(index='DayOfWeek', columns='Hour', values='Total')
    node_df_pivoted.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plot_circular_heatmap(node_df_pivoted, inner_r=0.1, colorbar_name='Total crossings', title="Hour", cmap='YlGn')


# noinspection PyUnusedLocal
def streamgraph_plot(data, col1, col2, **kwargs):
    """Draw a steamgraph plot of crossings, with col1 above and col2 below"""
    plot_df = data[[col1, col2, 'Time']].groupby('Time').mean()
    plot_df[col2] = -plot_df[col2]
    xticks = pd.date_range('00:00:00', '23:59:59', freq='240T').astype(str)
    ax = plt.gca()
    plot_df.plot(kind='area', ax=ax, xticks=xticks)
    ax.set(xlabel="Time", ylabel="Average crossings per hour")
    xtick_labels = [xtick.split(' ')[-1] for xtick in xticks]
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')


def plot_streamgraph_small_multiples(data):
    """Draw steamgraph plots for each day of the week in small multiples, with one direction above and
    the other below"""
    def make_plot(node, season):
        plot_data = data[data['Node'] == node]
        if season > 0:
            start_month_season = (season - 1) * 3 + 1
            end_month_season = season * 3
            plot_data = plot_data[plot_data['Date'].apply(lambda x: (x.month >= start_month_season) &
                                                                    (x.month <= end_month_season))]

        g = sb.FacetGrid(plot_data, col="DayOfWeek", col_wrap=5, sharex=False, legend_out=True, height=2.5, aspect=1)
        g.map_dataframe(streamgraph_plot, col1='Direction 0', col2='Direction 1')
        g = g.add_legend()

        g.fig.subplots_adjust(hspace=1, right=0.8)
        for _j, ax in enumerate(g.fig.get_axes()):
            ax.set_title(days_of_week[_j], y=1.2)
        plt.show()

    node_selector = widgets.Dropdown(options=reordered_nodes, value=reordered_nodes[0], description='Node')
    season_options = [('All year', 0), ('Winter (Jan-March)', 1), ('Spring (Apr-Jun)', 2),
                      ('Summer (Jul-Sept)', 3), ('Fall (Oct-Dec)', 4)]
    season_selector = widgets.Dropdown(options=season_options, value=0, description='Season')

    interact(make_plot, node=node_selector, season=season_selector)


def plot_results_pca(pivoted_data_per_node):
    """Plot the results of PCA for the pivoted data per node"""
    pivoted_values_per_node = pivoted_data_per_node.reset_index().drop(['Date', 'Node'], axis=1).values
    pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(pivoted_values_per_node), columns=['comp1', 'comp2'])
    g = sb.lmplot(data=pca_df, x='comp1', y='comp2', fit_reg=False)
    g.set(xticklabels=[], yticklabels=[], xlabel='', ylabel='')


def plot_pca_per_day(df):
    """Draw plot of the results of PCA with each day of the week in a different color, with interactive choice
     of the node and of the color map"""

    def make_plot(cmap, node):
        node_data = df[df['Node'] == node].dropna()

        pivoted_data = node_data.pivot_table(values='Total', index='Date', columns='Hour').dropna()
        pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(pivoted_data.values), columns=['comp1', 'comp2'])

        pca_df['DayOfWeek'] = pd.to_datetime(pivoted_data.index).dayofweek + 1

        pca_df = pca_df.sort_values(by='DayOfWeek')

        pca_df['DayOfWeek'] = pca_df['DayOfWeek'].apply(lambda x: days_of_week[x-1])

        color_list = sb.color_palette(cmap, 7)
        color_list = [f'rgb{c[0], c[1], c[2]}' for c in color_list]

        fig = px.scatter(pca_df, x='comp1', y='comp2', color='DayOfWeek', color_discrete_sequence=color_list)
        fig['layout']['xaxis'].update(range=[pca_df['comp1'].min() - 10, pca_df['comp1'].max() + 10])

        fig['layout']['yaxis'].update(range=[pca_df['comp2'].min() - 10, pca_df['comp2'].max() + 10])

        fig.show(renderer="colab")

    options = [(node, node) for node in reordered_nodes]
    controller_nodes = vpwt.get_controller({'widget': 'Dropdown',
                                            'options': options,
                                            'value': 'Fremont_Bridge',
                                            'description': 'Node'})

    cmaps = {'Sequential (YlGn)': 'YlGn', 'Divergent (RdBu)': 'RdBu', 'Qualitative (colorblind)': 'colorblind'}
    default = 'YlGn'
    # define selector
    controller_cmap = widgets.Dropdown(options=cmaps, value=default, description='Colormap')

    interact(make_plot, cmap=controller_cmap, node=controller_nodes)


def plot_pca_clusters(df):
    """Plot the results of clustering with a Gaussian Mixture Model after PCA"""
    def make_plot(node):
        node_data = df[df['Node'] == node].dropna()
        pivoted_for_pca = node_data.pivot_table(values='Total', index='Date', columns='Hour').dropna()
        pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(pivoted_for_pca.values),
                              columns=['comp1', 'comp2'])
        gmm = GaussianMixture(2, covariance_type='full', random_state=0)
        gmm.fit(pca_df)
        pca_df['label'] = gmm.predict(pca_df)

        p = sb.lmplot(data=pca_df, x='comp1', y='comp2', hue='label', fit_reg=False,
                      palette=sb.color_palette("Set1", 2))
        p = p.set_axis_labels('', '')
        p = p.set_xticklabels('')
        p.set_yticklabels('')
        plt.show()

    options = [(node, node) for node in reordered_nodes if node not in ['NW_58th_St', '26th_Ave']]

    controller_nodes = vpwt.get_controller({'widget': 'Dropdown',
                                            'options': options,
                                            'value': 'Fremont_Bridge',
                                            'description': 'Node'})

    interact(make_plot, node=controller_nodes)


def plot_heatmap_outliers(data, node='Fremont_Bridge'):
    """Draw a heatmap of the weekdays assigned to the weekend cluster after clustering using a Gaussian Mixture Model
    on the PCA data, showing whether the dates corresponded to a holiday"""
    node_data = data[data['Node'] == node].dropna()
    pivoted_for_pca = node_data.pivot_table(values='Total', index='Date', columns='Hour').dropna()
    pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(pivoted_for_pca.values), columns=['comp1', 'comp2'])
    pca_df['DayOfWeek'] = pd.to_datetime(pivoted_for_pca.index).dayofweek
    gmm = GaussianMixture(2, covariance_type='full', random_state=0)
    gmm.fit(pca_df)
    pca_df['label'] = gmm.predict(pca_df)
    pivoted_for_pca['cluster'] = pca_df['label'].values
    tmp = node_data.groupby(['Date', 'DayOfWeek']).sum().reset_index().join(pivoted_for_pca['cluster'], on='Date')
    weekdays_as_weekends = tmp[(tmp['cluster'] == 1) & (tmp['DayOfWeek'] < 6)]

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays('2012', '2019', return_name=True)
    weekdays_as_weekends = weekdays_as_weekends.set_index('Date').merge(holidays.to_frame(), left_index=True,
                                                                        right_index=True, how='left')[[0]]
    weekdays_as_weekends = weekdays_as_weekends.reset_index(drop=False)
    weekdays_as_weekends.columns = ['Date', 'Holiday']
    weekdays_as_weekends['Holiday'] = (weekdays_as_weekends.Holiday
                                       .fillna(weekdays_as_weekends.Date.apply(lambda x: x.strftime('%b %d'))))
    weekdays_as_weekends['Year'] = weekdays_as_weekends.Date.apply(lambda x: x.year)
    weekdays_as_weekends = weekdays_as_weekends.pivot_table(aggfunc='count', columns='Year', index='Holiday')
    weekdays_as_weekends = weekdays_as_weekends.fillna(0)
    weekdays_as_weekends = weekdays_as_weekends.astype(bool)
    weekdays_as_weekends.columns = [x[1] for x in weekdays_as_weekends.columns]
    sb.heatmap(weekdays_as_weekends, annot=weekdays_as_weekends.astype(str), fmt='s',
               cmap=sb.palettes.diverging_palette(15, 250, l=57, s=90, n=2), cbar=False)
