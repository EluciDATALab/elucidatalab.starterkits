# ©, 2022, Sirris
# owner: MDHN

import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import support as sp
import datetime as dt

from scipy.stats import pearsonr

sb.set()


def plot_train_delays(df, kind='day'):
    """
    Plot train delay distribution per day of the week or per month

    :param df. a pandas dataframe that contains OTP data
    :param kind. how to visualize the delays. available values are day and moth. Default = day
    """

    df = df.copy()
    df = sp.get_day_of_the_week(df=df)

    if kind.lower() == 'day':
        g = sb.boxplot(x='dayOfTheWeek', y='delay', data=df, showfliers=False)
        g.set_title('Train delay distribution per day of the week')

    elif kind.lower() == 'month':
        df['month'] = df.date.apply(lambda x: dt.date.strftime(x, '%b'))

        g = sb.barplot(x='month', y='delay', data=df)
        g.set_title('Train delay distribution per month')

    else:
        raise ValueError("Not a valid type")


def plot_train_density(df, percentile=75):
    """
    Visualize train density per hour on weekdays

    :param df. a pandas dataframe that contains OTP data
    :param percentile. percentile value to use for defining the threshold
    """

    df = df.copy()
    threshold = sp.define_threshold(df=df, percentile=percentile)
    train_density = sp.define_train_density(df=df)

    n = 23

    # create theta for 24 hours
    train_density['theta'] = np.linspace(0.0, 2 * np.pi, n, endpoint=False)

    # width of each bin on the plot
    width = (2 * np.pi) / n

    # make a polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.bar(train_density.theta, train_density.ct, width=width, bottom=2)
    rads = np.arange(0, (2 * np.pi), 0.01)

    plt.polar(rads, [threshold] * len(rads), '-', color='red')

    # set the labels go clockwise and start from the top
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    # set the labels
    ax.set_xticklabels(['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
    ax.set_title("# of distinct trains per hour across all weekdays (Mon-Fri)")

    plt.show()


def plot_delays_per_stop_number(df):
    """
    Plot delay as a function of the stop/rank number

    :param df. a pandas dataframe that contains OTP data
    """

    df = df.copy()

    matplotlib.rcParams['figure.figsize'] = (15, 6)
    f, axes = plt.subplots(1, 2)

    # plot delays per stop
    sb.boxplot(x='rank', y='delay', data=df.query("rank<=20"), ax=axes[0])
    axes[0].set_title('Boxplot distribution of delays per rank number')
    axes[0].set_ylabel('Delay (min)')
    axes[0].set_xlabel('Rank number')
    axes[0].set_xticks(np.arange(0, 20))
    axes[0].set_xticklabels(np.arange(1, 21))

    # calculate median delay per rank number
    delay_stop = df.groupby('rank').delay.median().reset_index(name='avg')
    sb.regplot(x='rank', y='avg', data=delay_stop, ax=axes[1])
    axes[1].set_title('Median delay vs rank number')
    axes[1].set_ylabel('Median delay (min)')
    axes[1].set_xlabel('Rank number')
    axes[1].set_ylim([0, delay_stop.avg.max()])


def plot_train_run(df_train_view, uni_coords, uni_runs, df_distance, df_coords, train_id='216', date='2016-10-07'):
    """
    Visualize train run

    :param df_train_view. a pandas dataframe with train view data
    :param uni_coords. combination of station/coordinates, 
    :param uni_runs. combination of of train_id/dates
    :param df_distance. coordinates for upcoming station 
    :param df_coords. coordinates for each train station per train_id and date as dataframes
    :param train_id. Train id used to select the train run
    :param date. date used to select the train run
    """

    df_train_view = df_train_view.copy()
    matplotlib.rcParams['figure.figsize'] = (10, 10)
    # identify and plot random example
    rand_run = df_coords.merge(uni_runs[(uni_runs.train_id == train_id) & (uni_runs.date == date)],
                               on=['train_id', 'date'])

    # define top destination stations to annotate plot with
    dest_count = df_train_view.groupby('dest').size().reset_index(name='ct')
    dest_count = dest_count[dest_count.ct > np.nanpercentile(dest_count.ct, 50)]
    edge_coords = df_distance[df_distance.next_station.isin(dest_count.dest.unique())][['next_station',
                                                                                        'lat',
                                                                                        'lon']].drop_duplicates()
    for k, i in edge_coords.iterrows():
        plt.annotate(i.next_station, (i.lon, i.lat))
        
    plt.scatter(uni_coords.lon, uni_coords.lat, marker='o', c='gray', alpha=0.4, s=100)
    plt.scatter(rand_run.lon, rand_run.lat, marker='o', c=rand_run['rank'], s=200)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    

def plot_distance_distribution(df):
    """
    Plot distribution of distances

    :param df. a pandas dataframe with distances
    """
    
    df = df.copy()
    matplotlib.rcParams['figure.figsize'] = (15, 5)
    ax = sb.histplot(df.distance, bins=100, kde=True, edgecolor=None, stat='density')
    ax.set(xlabel='Distance [km]')
    

def delay_jointplot(df, uni_runs):
    """
    jointplot to visualize delays as a function of distance between stations and cumulative distance of train run

    :param df. combination of station/coordinates
    :param uni_runs. combination of of train_id/dates
    """
    # subset for plotting efficiency
    df_coords_sub = df.merge(uni_runs.sample(1000), on=['train_id', 'date'])

    # plot delays per stop
    data1 = df_coords_sub.query("distance < 10")
    correlation_present1 = _is_correlation_present(data1['distance'], data1['delay'])
    g1 = sb.jointplot(x='distance', y='delay', data=data1, height=6,
                      kind='reg', fit_reg=correlation_present1,
                      scatter_kws={'alpha': 0.1}, line_kws={"color": "red"})
    g1.fig.suptitle('Delay as a function of distance between stations (limited to distances < 10km)')
    g1.set_axis_labels('Distance (km)', 'Delay (min)', fontsize=12)
    g1.fig.tight_layout()

    # calculate median delay per stop # and display
    correlation_present2 = _is_correlation_present(df_coords_sub['cum_distance'], df_coords_sub['delay'])
    g2 = sb.jointplot(x='cum_distance', y='delay', data=df_coords_sub, height=6,
                      kind='reg', fit_reg=correlation_present2,
                      scatter_kws={'alpha': 0.1}, line_kws={"color": "red"})
    g2.fig.suptitle('Delay as a function of cumulative distance of train run')
    g2.set_axis_labels('Cumulative distance (km)', 'Delay (min)', fontsize=12)
    g2.fig.tight_layout()


def total_train_run_distance_joint_plot(df):
    """
    regression plot to visualize total train run delay and delay at last stop as a function total run distance

    :param df. combination of station/coordinates
    """

    df = df.copy()

    # summarize by run
    df_run = df.groupby(['train_id', 'date']).agg({'distance': 'sum', 'delay': ['sum', 'last']})
    df_run.columns = ['distance', 'delay_sum', 'delay_last']

    # plot delays per stop
    correlation_present1 = _is_correlation_present(df_run['distance'], df_run['delay_sum'])
    g1 = sb.jointplot(x='distance', y='delay_sum', data=df_run, kind='reg', height=6,
                      fit_reg=correlation_present1,
                      scatter_kws={'alpha': 0.1}, line_kws={"color": "red"})
    g1.fig.suptitle('Total train run delay as a function total run distance')
    g1.set_axis_labels('Distance (km)', 'Delay (min)', fontsize=12)
    g1.fig.tight_layout()

    # calculate median delay per stop # and display
    correlation_present2 = _is_correlation_present(df_run['distance'], df_run['delay_sum'])
    g2 = sb.jointplot(x='distance', y='delay_last', data=df_run, kind='reg', height=6,
                      fit_reg=correlation_present2,
                      scatter_kws={'alpha': 0.1}, line_kws={"color": "red"})
    g2.fig.suptitle('Delay at last stop as a function total run distance')
    g2.set_axis_labels('Distance (km)', 'Delay (min)', fontsize=12)
    g2.fig.tight_layout()


def plot_northbound_southbound_dely_per_station(df):
    """
    Plot delay per station for northbound and southbound journeys

    :param df. a pandas dataframe with OTP data
    """

    df = df.copy()
    directions = pd.pivot_table(df, index=df.next_station, values='delay', columns='direction', aggfunc=np.mean)

    matplotlib.rcParams['figure.figsize'] = (7, 7)
    fig, axes = plt.subplots()
    sb.regplot(x='N', y='S', data=directions, fit_reg=False)
    axes.set_xlabel('Delay North direction (min)')
    axes.set_ylabel('Delay South direction (min)')
    axes.set_title('Delay per station for Northbound and Southbound journeys')
    ax_lims = [0, np.ceil(directions.max().max())]
    axes.set_ylim(ax_lims)
    axes.set_xlim(ax_lims)
    axes.add_line(plt.Line2D(ax_lims, ax_lims, color='black'))


def plot_stations_with_long_delays(df):
    """
    Barplots to show the stations with long delays for northbound and southbound directions

    :param df. a pandas dataframe with OTP data
    """

    df = df.copy()

    station_delay = sp.calculate_station_delay(df=df)

    # discard stations occurring less than the 5th percentile of occurrences
    station_delay = station_delay.query("ct > %f" % np.nanpercentile(station_delay.ct, 5))

    # sort and add rank
    station_delay = station_delay.reset_index().sort_values(['direction', 'avg'], ascending=[True, False])
    station_delay['rk'] = station_delay.groupby('direction').avg.rank(method='dense', ascending=False)

    # plot top 20 stations in each direction
    matplotlib.rcParams['figure.figsize'] = (14, 5)
    fig, ax = plt.subplots()
    top_delay = station_delay.query("rk < 21 & direction=='N'")
    sb.barplot(x='next_station', y='delay', order=top_delay.next_station,
               data=df.merge(top_delay, on=['direction', 'next_station']))
    ax.set_title('Direction: N')
    ax.set(ylabel='Delay (min)', xlabel='Train station')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        
    fig, ax = plt.subplots()
    top_delay = station_delay.query("rk < 21 & direction=='S'")
    sb.barplot(x='next_station', y='delay', order=top_delay.next_station,
               data=df.merge(top_delay, on=['direction', 'next_station']))
    ax.set_title('Direction: S')
    ax.set(ylabel='Delay (min)', xlabel='Train station')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


def plot_average_delay(df, percentile=90):
    """
    Plot average delay

    :param df. a pandas dataframe with OTP data
    :param percentile. percentile value to use for setting the threshold. Default = 90
    """

    df = df.copy()
    
    station_delay = sp.calculate_station_delay(df=df)
    delay_threshold = sp.calculate_delay_threshold(df=station_delay, percentile=percentile)
    matplotlib.rcParams['figure.figsize'] = (12, 6)

    fig, axes = plt.subplots()
    sb.distplot(station_delay.avg, ax=axes, kde=False)
    plt.axvline(delay_threshold, color='red')
    axes.set_ylabel('Count')
    axes.set_title('Average station delay')
    axes.set_xlabel('Average delay (min)')


def plot_track_changes(df):
    """
    Plot track changes

    :param df. a pandas dataframe with track changes
    """

    g = sb.boxplot(x="track_change_formatted", y="delay", data=df, showfliers=False, notch=True)
    g.set(ylabel='Delay (min)', xlabel='')
    g.set_xticklabels(['No track change', 'Track change'])

    
def _is_correlation_present(x, y, min_pearson_coef=0.3, verbose=0):
    correlation_present = True
    pearson_coef, p_value = pearsonr(x, y)
    
    if p_value >= 0.05 or abs(pearson_coef) < min_pearson_coef:
        correlation_present = False
    if verbose > 0:
        print(f'pearson_coef: {pearson_coef}, p_value: {p_value} ==> correlation_present: {correlation_present}')
    return correlation_present
    

def plot_track_change_frequency_with_delays(df_track_change_frequency, df_otp):
    """
    Plot how delay changes as a function of track change frequency

    :param df_track_change_frequency. a pandas dataframe with track change frequency
    :param df_otp. a pandas dataframe with OTP data
    """

    df_track_change_frequency = df_track_change_frequency.copy()
    df_otp = df_otp.copy()

    # add median delay per station to track changes
    track_change_delay = df_track_change_frequency.merge(
        df_otp.groupby(['next_station', 'date']).delay.mean().reset_index(name='delay'),
        on=['next_station', 'date'])

    correlation_present = _is_correlation_present(track_change_delay['track_change_frequency'],
                                                  track_change_delay['delay'])
    g = sb.jointplot(x='track_change_frequency', y='delay', data=track_change_delay, height=6,
                      kind='reg', fit_reg=correlation_present,
                     scatter_kws={'alpha': 0.1}, line_kws={"color": "red"})
    g.fig.suptitle('Track change frequency as a function of the mean delay')
    g.set_axis_labels('Track change frequency', 'Mean delay (min)', fontsize=12)
    g.fig.tight_layout()

    pearson_r = pearsonr(track_change_delay.delay,
                         track_change_delay.track_change_frequency)
    print(f"Pearson's coefficient = {pearson_r[0]:0.2f} with p-value = {pearson_r[1]:0.2e}.")


def plot_highest_track_change_frequency(track_change_frequency, n_stations=10):
    """
    Plot stations with the highest track change frequency

    :param track_change_frequency. a pandas dataframe with track change frequency records
    :param n_stations. Number of stations to plot
    """
    plt.figure(figsize=(12, 5))
    g = sb.barplot(x='next_station', y='track_change_frequency', data=track_change_frequency.head(n_stations))
    g.set_xlabel('')
    g.set_ylabel('Track change frequency')
    for tick in g.get_xticklabels():
        tick.set_rotation(45)


def plot_lowest_track_change_frequency(track_change_frequency, n_stations=10):
    """
    Plot stations with the lowest track change frequency

    :param track_change_frequency. a pandas dataframe with track change frequency records
    :param n_stations. Number of stations to plot
    """
    plt.figure(figsize=(12, 5))
    g = sb.barplot(x='next_station', y='track_change_frequency',
                   data=track_change_frequency.tail(n_stations))
    g.set_xlabel('')
    g.set_ylabel('Track change frequency')
    for tick in g.get_xticklabels():
        tick.set_rotation(45)
