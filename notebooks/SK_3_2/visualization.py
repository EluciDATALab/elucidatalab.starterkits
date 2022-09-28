import seaborn as sns
import matplotlib.pyplot as plt
from support import *
import folium
from elucidata_demonstrator_3_2 import (
    assert_correct_package_versions,
    Map,
    plot_barplot_with_labels,
    plot_double_barplot,
    extract_destination_station_frequency_per_gender,
)


def plot_difference_between_morning_evening_arrival(df):
    m = Map()
    m.add_markers(get_trips_top_arrival_morning_rush(df), popup_name='to_station_name',
                  marker=folium.Circle, lat='to_station_lat', lon='to_station_lon', radius='count', color='green')
    m.add_markers(get_trips_top_arrival_evening_rush(df), popup_name='to_station_name',
                  marker=folium.Circle, lat='to_station_lat', lon='to_station_lon', radius='count', color='red')

    return m


def plot_unbalanced_stations(df, n_stations=10):
    m = Map()
    df_topUnbalanced_stations = get_unbalanced_stations(df, n_stations=n_stations)
    m.add_markers(df_topUnbalanced_stations[df_topUnbalanced_stations['TripDifference'] > 0],
                  marker=folium.Circle ,radius='AbsTripDifference', popup_name='station_name', color='green')
    m.add_markers(df_topUnbalanced_stations[df_topUnbalanced_stations['TripDifference'] <= 0],
                  marker=folium.Circle, radius='AbsTripDifference', popup_name='station_name', color='red')

    return m


def plot_trip_duration(df):
    """
    Plots the distribution of the trip duration

    Args:
        df: A dataframe

    Returns:
        An axis

    """
    fig, ax = plt.subplots(figsize=(16, 6))
    g = sns.distplot(df.tripdurationMinutes, kde=False, bins=200, ax=ax)
    g.set_xlabel('Trip duration (minutes)', fontsize=16)
    g.set_ylabel('Number of trips', fontsize=16)
    g.tick_params(axis='both', which='major', labelsize=14)
    g.set_title('Distribution of Trip Duration Values (in minutes)', fontsize=16)
    g.set(yscale="log");

    print(f"The longest trip lasts: {round(df.tripdurationMinutes.max() / 60)} hours")

    return ax


def plot_user_age(df):
    """
    Plot the distribution of the user ages
    Args:
        df: A dataframe
    """
    plt.figure(figsize=(15, 5))
    g = sns.distplot(df.age.dropna(), kde=False, bins=np.arange(df.age.max()))

    age_mean = round(df.age.mean())
    age_mode = round(df.age.mode().values[0], 0)
    g.axvline(age_mean, color='red', linestyle='--')
    g.axvline(age_mode, color='green', linestyle='--')
    g.set_title(
        f"Distribution of Users' Ages\n"
        f"Green and red vertical lines indicate mode ({age_mode}) and mean ({age_mean}), "
        f"respectively.");


def plot_user_birth_year(df):
    """
    Plot the distribution of the users birth year

    Args:
        df: A dataframe
    """
    plt.figure(figsize=(15, 5))
    g = sns.distplot(df.birthyear.dropna(), kde=False);
    g.axvline(1987, color='green', linestyle='--')
    g.axvline(1985, color='red', linestyle='--')
    g.set_title("Distribution of Users' Ages\n"
                "Green and red vertical lines are located in the years 1985 and 1987, respectively.");


def plot_trip_count_per_gender(df):
    """
    Plot the number of trips per gender

    Args:
        df: A dataframe
    """
    gender_count = get_gender_count(df)
    plot_barplot_with_labels(gender_count, x='gender', y='count', labels='count', ymax=80000,
                             title='Trip Count per Gender');


def plot_station_ranks_by_popularity(df, n_stations=10):
    """
    Plot stations rank based on their popularity

    Args:
        df: A dataframe
        n_stations: Number of stations to plot
    """
    stations_ordered_by_count = get_stations_ordered_by_count(df)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Top {n_stations} Most and Least Popular Stations of Arrival')
    ylim = stations_ordered_by_count['count'].max()
    for ax, rank in zip(axes, ['top', 'bottom']):
        stations = stations_ordered_by_count[:n_stations] if rank == 'top' else \
            stations_ordered_by_count[-n_stations:]
        g = sns.barplot(x='to_station_name', y='count', data=stations, ax=ax)
        ax.set_xlabel('Station of Arrival')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontdict={'horizontalalignment': 'right'})
        ax.set_ylim((0, ylim));
        ax.set_title(f'Top {n_stations} Stations' if rank == 'top' else f'Bottom {n_stations} Stations')


def plot_trips_per_hour(df):
    week_days = get_weekdays()
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(18, 9))
    fig.suptitle('Number of Trips Per Hour for Weekdays (left) and Weekends (right)')
    for ax, w in zip(axes, ['weekday', 'weekend']):
        # df_trip_hour = (df_trips[df_trips.day.isin(week_days[w])]
        #     .groupby('hour')
        #     .size()
        #     .reset_index(name='ct'))
        # df_trip_hour.ct = df_trip_hour.ct / len(week_days[w])  # normalize by number of days

        df_trip_hour = get_trip_hour(df, week_days[w])

        # make a polar plot
        df_trip_hour['theta'] = np.linspace(0.0, 2 * np.pi, len(df_trip_hour), endpoint=False)  # 24 hours
        width = (2 * np.pi) / len(df_trip_hour)  # width of each bin on the plot
        bars = ax.bar(df_trip_hour.theta, df_trip_hour.ct, width=width, bottom=2)
        ax.set_theta_direction(-1)  # set the labels, go clockwise and start from the top
        ax.set_theta_zero_location("N")
        ax.set_ylim(0, 2300)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
        ax.set_xticklabels(range(24))
        ax.set_title(w.title());


def plot_station_of_arrivals_in_rush_hours(df):
    df_trips_topArrival_morning_rush = get_trips_top_arrival_morning_rush(df)
    df_trips_topArrival_evening_rush = get_trips_top_arrival_evening_rush(df)
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='col', figsize=(15, 5))
    df1 = df_trips_topArrival_morning_rush[['to_station_name', 'count']].head(10)
    df2 = df_trips_topArrival_evening_rush[['to_station_name', 'count']].head(10)
    ax0 = plot_barplot_with_labels(df1, x='to_station_name', y='count', labels='count',
                                   title='Top ten of stations of arrival in the morning rush hours',
                                   ymax=2000, ax=ax[0])
    ax0.axhline(1400, c='k', ls='--')
    ax1 = plot_barplot_with_labels(df2, x='to_station_name', y='count', labels='count',
                                   title='Top ten of stations of arrival in the evening rush hours',
                                   ymax=2000, ax=ax[1])
    ax1.axhline(1400, c='k', ls='--');


def plot_trip_duration_age_joint_distribution(df):
    df_trips_complete = df.dropna()
    pearson_corr = np.corrcoef(df_trips_complete.age, df_trips_complete.tripdurationMinutes)[0, 1]

    g = sns.jointplot(x='age', y='tripdurationMinutes', data=df.reset_index().drop_duplicates(subset=['index']),
                      kind='reg', height=6, scatter_kws={'alpha': 0.1})
    g.fig.suptitle('Joint Distribution of Trip Duration and Age')
    print(f"Trip duration (in minutes) vs age: Pearson correlation: {pearson_corr:0.2f}")


def plot_gender_influence_on_trip_duration(df):
    trip_duration_gender = get_trip_duration_gender(df)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    sns.heatmap(pd.DataFrame(trip_duration_gender["Number of Trips"]), ax=axes[0])
    sns.heatmap(pd.DataFrame(trip_duration_gender["Trip Duration in Minutes"]), ax=axes[1])


def plot_age_distribution_per_gender(df):
    df_tripAge = df[~df.age.isnull()]
    bins = list(range(0, 81))
    g = sns.FacetGrid(data=df_tripAge, hue="gender", height=5, aspect=4, legend_out=False,
                      hue_order=["Male", "Female", "Other"])
    g = g.map(sns.distplot, "age", bins=bins, hist_kws=dict(alpha=1), kde=False).add_legend()
    g.axes[0, 0].set(ylabel='Count', xlim=[15, 80])
    g.fig.suptitle('Age Distribution Per Gender');


def plot_trips_over_days(df):
    g = sns.factorplot(x='day', data=df, size=5, aspect=2, kind='count', hue='usertype')
    g.set(ylabel='number of trips')
    g.fig.suptitle('Number of Trips per Day and per User Type');


def plot_gender_over_days(df):
    df_trips_gender = df.groupby(['day', 'gender']).size().reset_index(name='ct')
    df_trips_gender['ratio'] = df_trips_gender.groupby('day')['ct'].transform(lambda x: x / x.sum() * 100)

    fig, ax = plt.subplots(figsize=(15, 5))
    bottom = [0] * df_trips_gender.day.nunique()
    for gender, color in zip(df_trips_gender.gender.unique(), sns.color_palette('Dark2')):
        df_trips_gender_sub = df_trips_gender[df_trips_gender.gender == gender].sort_values('day')
        ax.bar(df_trips_gender_sub.day, df_trips_gender_sub.ratio, bottom=bottom, color=color, label=gender)
        bottom = bottom + df_trips_gender_sub.ratio.values
    ax.set_ylabel('%')
    ax.set_xlabel('Day of the Week')
    ax.set_title('Percentage of Users Per Gender for Each Day of the Week')
    ax.legend();


def plot_stations_popularity_per_gender(df):
    df_arrivalStationFrequency_wm = extract_destination_station_frequency_per_gender(df)
    df_arrivalStationFrequency_wm_top10 = (df_arrivalStationFrequency_wm[['ratio_f2m', 'ratio_m2f']]
        .sort_values([('ratio_f2m')], ascending=False)
        .iloc[np.r_[0:5, -5:0]])
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_double_barplot(df_arrivalStationFrequency_wm_top10.reset_index(), ax, 'Station popularity per gender')
    plt.show()


def plot_trips_per_station(df):
    df_trips_stations = get_trips_per_stations(df)

    fig, ax = plt.subplots()
    sns.scatterplot(x='from_station', y='to_station', data=df_trips_stations, s=100, ax=ax)
    t_max = df_trips_stations[['to_station', 'from_station']].max().max()
    ax.plot([0, t_max], [0, t_max], ls="--", c=".3")
    ax.set(
        xlim=[0, None],
        xlabel='Origin',
        ylim=[0, None],
        ylabel='Destination',
        title='Count of Trips per Station when it is the Trip Origin (x-axis) and Destination (y-axis).');


def plot_trip_duration_over_time(df, time='both', axes=None):
    """
    Plot trip duration over time

    Args:
        df: A dataframe
        time: The possible values are day, month, or both
        axes: An axis
    """
    df_trips_dow = get_trips_dow(df)
    df_trips_moy = get_trips_moy(df)

    if axes is None:
        if time.lower() == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        elif (time.lower() == 'day') | (time.lower() == 'month'):
            fig, axes = plt.subplots(figsize=(9, 8))

    if time.lower() == 'both':
        axes[0].set_title("Number and Median Duration (in Minutes)\n of Trips Per Day of the Week", size=19)
        sns.barplot(x='day', y='n_trips', data=df_trips_dow, ax=axes[0], dodge=False)
        axes[0].set_xlabel('Day of the Week', size=14)
        axes[0].set_xticklabels(df_trips_dow.day)
        axes[0].set_ylabel('Number of Trips', size=14)
        ax2 = axes[0].twinx()
        sns.lineplot(x='day', y='tripdurationMinutes', data=df_trips_dow, axes=ax2, marker='o', color='black')
        ax2.set_ylim([0, 30])
        ax2.set_ylabel('Median Trip Duration (Minutes)', size=14);

        axes[1].set_title("Number and Median Duration (in Minutes)\n of Trips Per Month", size=19)
        sns.barplot(x='month', y='n_trips', data=df_trips_moy, ax=axes[1], dodge=False)
        axes[1].set_xlabel('Month', size=14)
        axes[1].set_xticklabels(df_trips_moy.month)
        axes[1].set_ylabel('Number of Trips', size=14)
        ax3 = axes[1].twinx()
        sns.lineplot(x='month', y='tripdurationMinutes', data=df_trips_moy, axes=ax3, marker='o', color='black')
        ax3.set_ylim([0, 30])
        ax3.set_ylabel('Median Trip Duration (Minutes)', size=14);
        plt.tight_layout();

    elif time.lower() == 'day':
        axes.set_title("Number and Median Duration (in Minutes)\n of Trips Per Day of the Week", size=19)
        sns.barplot(x='day', y='n_trips', data=df_trips_dow, ax=axes, dodge=False)
        axes.set_xlabel('Day of the Week', size=14)
        axes.set_xticklabels(df_trips_dow.day)
        axes.set_ylabel('Number of Trips', size=14)
        ax2 = axes.twinx()
        sns.lineplot(x='day', y='tripdurationMinutes', data=df_trips_dow, axes=ax2, marker='o', color='black')
        ax2.set_ylim([0, 30])
        ax2.set_ylabel('Median Trip Duration (Minutes)', size=14);

    elif time.lower() == 'month':
        axes.set_title("Number and Median Duration (in Minutes)\n of Trips Per Month", size=19)
        sns.barplot(x='month', y='n_trips', data=df_trips_moy, ax=axes, dodge=False)
        axes.set_xlabel('Month', size=14)
        axes.set_xticklabels(df_trips_moy.month)
        axes.set_ylabel('Number of Trips', size=14)
        ax3 = axes.twinx()
        sns.lineplot(x='month', y='tripdurationMinutes', data=df_trips_moy, axes=ax3, marker='o', color='black')
        ax3.set_ylim([0, 30])
        ax3.set_ylabel('Median Trip Duration (Minutes)', size=14);
        plt.tight_layout();


def plot_trip_duration_difference_between_stations(df):
    df_trip_duration = get_unbalanced_trip_duration(df)
    trip_duration = pd.pivot_table(df_trip_duration,
                                   index='from_station',
                                   columns='to_station',
                                   values='median duration (min)')
    stations_sort = df_trip_duration.sort_values('count', ascending=True).from_station.unique()
    trip_duration = trip_duration.loc[stations_sort, stations_sort]
    nof_dest = len(trip_duration)
    tri_upper = np.triu(trip_duration, k=1).reshape(nof_dest, nof_dest)
    tri_lower = np.tril(trip_duration, k=-1).reshape(nof_dest, nof_dest)

    distance_difference_from_to = pd.DataFrame(tri_lower - tri_upper.T)
    distance_difference_from_to.set_axis(trip_duration.columns.values, axis=1, inplace=True)
    distance_difference_from_to.set_axis(trip_duration.columns.values, axis=0, inplace=True)

    # colored-Matrix
    mask = np.zeros_like(distance_difference_from_to, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(11, 9))
    g = sns.heatmap(distance_difference_from_to, mask=mask, cmap=cmap, vmax=15, vmin=-15)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, fontdict={'horizontalalignment': 'right'})
    g.set_ylabel('from station')
    g.set_xlabel('to station')
    g.set_title('Difference in Trip Duration (min) Between Stations');

    # Stations located in Capitol Hill
    for i in range(5, 10):
        g.get_xticklabels()[i].set_weight("bold")
        g.get_yticklabels()[i].set_weight("bold")


def plot_arrival_station_frequency_wm_weekend(df, gender='both'):
    df_arrivalStationFrequency_wm_weekend_top10women = get_arrival_station_frequency_wm_weekend_women(df, n_stations=10)
    df_arrivalStationFrequency_wm_weekend_top10men = get_arrival_station_frequency_wm_weekend_men(df, n_stations=10)

    if gender.lower() == 'women':
        fig, ax = plt.subplots(figsize=(8, 5))

        df = df_arrivalStationFrequency_wm_weekend_top10women[['to_station_name', 'ratio_f2m', 'ratio_m2f']]
        plot_double_barplot(df, ax, 'Top ten arrival stations during weekend for women')
        plt.show()

    elif gender.lower() == 'men':
        fig, ax = plt.subplots(figsize=(8, 5))

        df = df_arrivalStationFrequency_wm_weekend_top10men[['to_station_name', 'ratio_f2m', 'ratio_m2f']]
        plot_double_barplot(df, ax, 'Top ten arrival stations during weekend for men')
        plt.show()

    elif gender.lower() == 'both':
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey='col', figsize=(15, 5))

        df1 = df_arrivalStationFrequency_wm_weekend_top10women[['to_station_name', 'ratio_f2m', 'ratio_m2f']]
        df2 = df_arrivalStationFrequency_wm_weekend_top10men[['to_station_name', 'ratio_f2m', 'ratio_m2f']]
        plot_double_barplot(df1, ax[1], 'Top ten arrival stations during weekend for women')
        plot_double_barplot(df2, ax[0], 'Top ten arrival stations during weekend for men')
        plt.show()


def plot_stations_of_arrival_on_weekdays_during_morning_rush_hour(df):
    fig, ax = plt.subplots(figsize=(15, 5))

    df_arrivalStationFrequency_wm_morningRushHour = get_arrival_station_frequency_wm_morning_rush_hour(df)

    df = (df_arrivalStationFrequency_wm_morningRushHour[['to_station_name', 'ratio_f2m', 'ratio_m2f']]
          .sort_values('ratio_m2f'))
    plot_double_barplot(df, ax, 'Top ten stations of arrival on weekdays during the morning rush hour.')
    plt.show()


def plot_stations(df, n_stations=10):
    m = Map()

    df_arrivalStationFrequency_wm_morningRushHour = get_arrival_station_frequency_wm_morning_rush_hour(df)
    df_arrivalStationFrequency_wm_weekend_top10women = get_arrival_station_frequency_wm_weekend_women(df=df,
                                                                                                      n_stations=n_stations)
    df_arrivalStationFrequency_wm_weekend_top10men = get_arrival_station_frequency_wm_weekend_men(df=df,
                                                                                                  n_stations=n_stations)

    m.add_markers(df_arrivalStationFrequency_wm_morningRushHour, color='blue',
                  marker=folium.Circle, lat='to_station_lat', lon='to_station_lon', popup_name='to_station_name')
    m.add_markers(df_arrivalStationFrequency_wm_weekend_top10women, color='green',
                  marker=folium.Circle, lat='to_station_lat', lon='to_station_lon', popup_name='to_station_name')
    m.add_markers(df_arrivalStationFrequency_wm_weekend_top10men, color='red',
                  marker=folium.Circle, lat='to_station_lat', lon='to_station_lon', popup_name='to_station_name')

    return m
