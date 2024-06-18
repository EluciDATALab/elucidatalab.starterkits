# ©, 2022, Sirris
# owner: MDHN

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import datetime as dt

from geopy.distance import geodesic
from starterkits import pipeline


warnings.filterwarnings('ignore')


def load_otp_data(force=False, DATA_PATH='../../data'):
    """Extract OTP dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally. The OTP dataset provides information about trains and
    the times at which they arrive at the different stations on their lines.
    
    :param force: if True, force re-download of data from server
    :param DATA_PATH: local path to data folder

    :return a pandas dataframe with OTP data
    """

    def extract_csv_file(file_name):
        target_folder = DATA_PATH + '/SK_3_3'
        zip_file_name = pipeline.download_from_repo(
            'SK_3_3',
            'otp.csv.zip',
            str(target_folder + '/otp.csv.zip'),
            force=force)

        with zipfile.ZipFile(zip_file_name, 'r') as zipped_file:
            zipped_file.extract('otp.csv', path=str(target_folder))
        os.remove(zip_file_name)
        return file_name

    f_name = (pipeline.File(extract_csv_file,
                            str(DATA_PATH + '/SK_3_3/otp.csv'))
              .make(force=force))

    df_otp = pd.read_csv(f_name)
    df_otp['date'] = df_otp.date.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    df_otp['timeStamp'] = df_otp.timeStamp.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    return df_otp


def load_train_view_data(force=False, DATA_PATH='../../data'):
    """Extract train_view dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally. The OTP dataset providestrain tracking information.

    :param force: if True, force re-download of data from server
    :param DATA_PATH: local path to data folder

    :return a pandas dataframe with train view data
    """

    def extract_csv_file(file_name):
        target_folder = DATA_PATH + '/SK_3_3'
        zip_file_name = pipeline.download_from_repo(
            'SK_3_3',
            'trainView.csv.zip',
            str(target_folder + '/trainView.csv.zip'),
            force=force)

        with zipfile.ZipFile(zip_file_name, 'r') as zipped_file:
            zipped_file.extract('trainView.csv', path=str(target_folder))
        os.remove(zip_file_name)
        return file_name

    f_name = (pipeline.File(extract_csv_file,
                            str(DATA_PATH + '/SK_3_3/trainView.csv'))
              .make(force=force))

    df_train_view = pd.read_csv(f_name)
    df_train_view['date'] = df_train_view.date.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    return df_train_view


def remove_invalid_data_instances(df):
    """
    Remove invalid data instances from OTP dataset

    :param df. a pandas dataframe with OTP data

    :return a pandas dataframe in which the invalid instances are removed
    """

    df = df.copy()

    mask = df.train_id.str.contains('[^0-9]+')
    df_weird_ids = df[mask]

    print('We found {} IDs with non-numeric characters, which is {}% from a total of {} IDs.'
          .format(df_weird_ids.index.size,
                  round(float(df_weird_ids.index.size) / df.index.size * 100, 2),
                  df.index.size))
    print('In addition, there are %d entries with next_station name = None' % (df.next_station == "None").sum())

    print('These entries with an unexpected train ID or no station name information will be excluded from the dataset')
    df = df[~mask]
    df = df[df.next_station != 'None']

    return df


def get_train_run(df):
    """
    a function that count how many time a train_id passes through a given station on a given day

    :param df. a pandas dataframe 

    :return a pandas dataframe
    """

    df = df.copy()
    df = df.groupby(['train_id', 'next_station', 'date']).size().reset_index(name='n_occurrences')
    df['single_pass'] = df.n_occurrences == 1

    return df


def exclude_remaining_cases(otp, train_view):
    """
    Exclude remaining 0.2% of cases that correspond to exceptional situations from otp and train_view datasets

    :param otp. a pandas dataframe with otp data
    :param train_view. a pandas dataframe with train_view data

    :return otp and train_view dataset with exceptional cases removed from the datasets
    """

    df_otp = otp.copy()
    df_train_view = train_view.copy()
    train_run = get_train_run(df=df_otp)
    df_otp = df_otp.merge(train_run[train_run.single_pass == True], on=['train_id', 'next_station', 'date'])
    df_train_view = df_train_view.merge(train_run[train_run.single_pass == True],
                                        on=['train_id', 'next_station', 'date'])

    return df_otp, df_train_view


def calculate_delays_for_otp(df):
    """
    Transform status column values into integers (in minutes) and rename the column to delay

    :param df. a pandas dataframe with otp data

    :return a pandas dataframe with calculated delays
    """

    df = df.copy()

    df = df[df.status != 'None']
    df['delay'] = df.status.str.replace(' min', '')
    df['delay'] = df['delay'].str.replace('On Time', '0')
    df['delay'] = df['delay'].astype(int)

    print(df[['status', 'delay']].head())
    df.drop('status', axis=1, inplace=True)

    return df


def calculate_delays_for_train_view(df):
    """
    Transform status column values into integers (in minutes) and rename the column to delay

    :param df. a pandas dataframe with train_view data

    :return a pandas dataframe with calculated delays
    """

    df = df.copy()
    df = df[df.status != 'None']
    df.status = df.status.astype(int)

    df = df.rename(columns={'status': 'delay'})

    return df


def remove_suspended_trains(df_otp, df_train_view):
    """
    Remove suspended trains

    :param df_otp. a pandas dataframe that contains otp data
    :param df_train_view. a pandas dataframe that contains df_train_view data

    :return two pandas dataframe (df_otp and df_train_view) in which the suspended trains removed
    """

    df_otp = df_otp.copy()
    df_train_view = df_train_view.copy()

    df_otp = df_otp[df_otp.delay != 999]
    df_train_view = df_train_view[df_train_view.delay != 999]

    df_otp = df_otp[df_otp.delay != 1440]
    df_train_view = df_train_view[df_train_view.delay != 1440]

    return df_otp, df_train_view


def get_otp_overview(df):
    """
    Show an overview of the OTP cleaned dataset

    :param df. a pandas dataframe that contains OTP data

    :return a pandas dataframe that contains OTP overview
    """
    
    df = df.copy()
    overview = {'Number of stations': df.next_station.nunique(),
                'Number of train ids': df.train_id.nunique(),
                'Number of train runs': df[['train_id', 'date']].drop_duplicates().size,
                'Time range': '%s to %s' % (dt.datetime.strftime(df.date.min(),
                                                                 '%d %b %Y'),
                                            dt.datetime.strftime(df.date.max(),
                                                                 '%d %b %Y')),
                'Percentage trains on time': '%0.1f%%' % (sum(df.delay < 6) / len(df) * 100)}
    overview = pd.DataFrame(overview, index=[0]).T
    overview.columns = ['']
    
    return overview


def get_day_of_the_week(df):
    """
    Get day of the week

    :param df. a pandas dataframe that contains OTP data

    :return a pandas dataframe that contains OTP data with a new column called day of the week
    """

    df = df.copy()
    df['dayOfTheWeek'] = df.timeStamp.apply(lambda x: dt.datetime.strftime(x, '%a'))
    df['dayOfTheWeek'] = pd.Categorical(df.dayOfTheWeek,
                                        categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    return df


def get_type_of_days(df):
    """
    Get the type of days

    :param df. a pandas dataframe that contains OTP data

    :return a pandas dataframe that contains OTP data with type of day
    """

    df = df.copy()
    df = get_day_of_the_week(df=df)

    wd = {w: 'weekday' for w in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']}
    wd['Sat'] = 'Sat'
    wd['Sun'] = 'Sun'
    df['typeOfDay'] = df.dayOfTheWeek.apply(lambda x: wd[x])

    return df


def extract_time_features(df):
    """
    Extract time features from OTP dataset

    :param df. a pandas dataframe that contains OTP data

    return a pandas dataframe that contains OTP data and extracted time features
    """

    df = df.copy()

    df['hour'] = df.timeStamp.apply(lambda x: dt.datetime.strftime(x, '%H'))
    df['isWeekend'] = df.dayOfTheWeek.isin(['Sat', 'Sun'])

    return df


def define_threshold(df, percentile=75):
    """
    Define rush hour

    :param df. a pandas dataframe that contains 
    :param percentile. percentile value to use for defining the threshold

    :return rush hour
    """

    return np.nanpercentile(df[df.isWeekend == False].groupby(['dayOfTheWeek', 'hour'])
                            .train_id.apply(lambda x: len(set(x))), percentile)


def define_train_density(df):
    """
    Define train density per hour

    :param df. a pandas dataframe that contains OTP data

    :return train density per hour
    """

    df = df.copy()
    train_density = df[df.isWeekend == False].groupby('hour').train_id.apply(lambda x: len(set(x))).reset_index(name='ct')
    return train_density  


def define_rush_hour(df, percentile=75):
    """
    Define rush hour

    :param df. a pandas dataframe that contains OTP data
    :param percentile. percentile value to use for defining the threshold
    
    :return a dataframe that contains OTP data and rush hour
    """

    df = df.copy()
    threshold = define_threshold(df=df, percentile=percentile)
    train_density = define_train_density(df=df)

    print("The threshold that we will use is of %0.0d trains per hour over all weekdays and hours." % threshold)

    rush_hours = train_density[train_density.ct > threshold].hour
    print("Identified rush hours are: %s." % ', '.join([str(hour) for hour in rush_hours]))
    df['isRushHour'] = (df.isWeekend == False) & (df.hour.isin(rush_hours))

    return df   


def calculate_rank_stop(df):
    """
    Calculate rank of the stops

    :param df. a pandas dataframe that contains OTP data

    :return a pandas dataframe that contains OTP data and the ranking of the stops
    """

    df = df.copy()
    df['rank'] = df.groupby(['train_id', 'date']).timeStamp.rank(ascending=True, method='dense')

    return df                


def calculate_distance_between_stations(df_otp, df_train_view):
    """
    Calculate the distance between the stations in km.

    :param df_otp. a pandas dataframe that contains otp data
    :param df_train_view. a pandas dataframe that contains df_train_view data

    :return combination of station/coordinates, combination of of train_id/dates, coordinates for upcoming station, 
        and coordinates for each train station per train_id and date as dataframes
    """

    df_otp = df_otp.copy()
    df_train_view = df_train_view.copy()

    # create a dataset with coordinates for each train station per train_id and date
    df_coords = df_otp[['train_id', 'date', 'rank', 'next_station', 'delay']].merge(
        df_train_view.groupby('next_station').agg({'lon': 'mean', 'lat': 'mean'}).reset_index(),
        on='next_station')

    # add also destination to the dataset
    df_coords = df_coords.merge(df_train_view[['train_id', 'date', 'dest', 'source']].drop_duplicates(),
                                on=['train_id', 'date'])

    # get unique combination of station/coordinates and of train_id/dates
    uni_coords = df_coords[['next_station', 'lon', 'lat']].drop_duplicates()
    uni_runs = df_coords[['train_id', 'date']].drop_duplicates()

    # shift next_station column to align in each row the current station and the upcoming one
    df_coords = df_coords.sort_values('rank')
    df_coords['upcoming_station'] = df_coords.groupby(['train_id', 'date']).next_station.shift(1)

    # add coordinates for upcoming station
    df_distance = df_coords[['next_station', 'lat', 'lon', 'upcoming_station']].drop_duplicates()\
        .merge(
            uni_coords.rename(columns={'lon': 'next_lon', 'lat': 'next_lat', 'next_station': 'upcoming_station'}),
            on='upcoming_station')\
        .query("next_station!=upcoming_station")

    # calculate distance
    df_distance['distance'] = df_distance.apply(lambda x: geodesic((x['lat'], x['lon']),
                                                                   (x['next_lat'], x['next_lon'])
                                                                   ).km, axis=1)

    # add distance to df_coords
    df_coords = df_coords.merge(df_distance[['next_station', 'upcoming_station', 'distance']],
                                on=['next_station', 'upcoming_station'])

    return uni_coords, uni_runs, df_distance, df_coords           


def calculate_cumulative_distance_along_train_run(df, uni_runs):
    """
    Calculate cumulative distance alonge a train run

    :param df. combination of station/coordinates
    :param uni_runs. combination of of train_id/dates

    :return a pandas dataframe with cumulative distance calculated
    """

    df = df.copy()

    df = df.sort_values('rank')
    df['cum_distance'] = df.groupby(['train_id', 'date']).distance.cumsum()

    # show random train run
    random_train_run = df.merge(uni_runs.sample(1), on=['train_id', 'date'])[['train_id', 'rank',
                                                                              'next_station', 'delay',
                                                                              'upcoming_station', 'distance',
                                                                              'cum_distance']]

    return df, random_train_run


def add_cum_delay(df):
    """
    Calculate cumulative delay

    :param df. a pandas dataframe with otp data

    :return a pandas dataframe with cumulative delay
    """

    df = df.copy()
    df = df.sort_values('rank')
    df['cum_delay'] = df.groupby(['train_id', 'date']).delay.cumsum()

    return df


def add_distance(df_otp, df_coords):
    """
    Calculate distance and cumulative distance

    :param df_otp. a pandas dataframe with otp data
    :param df_coords. combination of station/coordinates

    :return a pandas dataframe with calculated distance
    """

    df_otp = df_otp.copy()
    df_coords = df_coords.copy()

    df_otp = df_otp.sort_values('rank')
    df_otp = df_otp.merge(df_coords[['train_id', 'date', 'next_station', 'distance', 'cum_distance']],
                          on=['train_id', 'date', 'next_station'])

    return df_otp


def calculate_station_delay(df):
    """
    Calculate station delay

    :param df. a pandas dataframe with OTP data
    """

    df = df.copy()

    station_delay = df.groupby(['direction', 'next_station']).delay.agg(avg='mean', stdev='std', ct='count')
    station_delay.columns = ['avg', 'stdev', 'ct']

    return station_delay


def calculate_delay_threshold(df, percentile=90):
    """
    Calculate delay threshold

    :param df: a pandas dataframe station delays
    :param percentile: percentile value to use for setting the threshold. Default: 90
    :return: delay threshold
    """

    delay_threshold = np.nanpercentile(df.avg, percentile)

    return delay_threshold


def calculate_long_delay_stations(station_delay, delay_threshold):
    """
    Calculate long delay stations

    :param station_delay. staions delays
    :param delay_threshold. delay thresholds

    :return stations with long delays
    """

    long_delay_station = station_delay.query("avg > %f" % delay_threshold)

    return long_delay_station


def print_delays_overview(df, percentile=90):
    """
    Give an overview of delay threshold and long delay stations

    :param df: a pandas dataframe with OTP data
    :param percentile: percentile value to use for setting the threshold. Default: 90
    """

    station_delay = calculate_station_delay(df=df)
    delay_threshold = calculate_delay_threshold(df=station_delay, percentile=percentile)
    long_delay_station = calculate_long_delay_stations(station_delay=station_delay, delay_threshold=delay_threshold)

    print("Stations with an average delay above %0.1f minutes will be labeled:" % delay_threshold)
    for direction in ['N', 'S']:
        print("\tDirection %s: %s" % (direction,
                                      ', '.join(long_delay_station.query("direction == '%s'" % direction
                                                                         ).reset_index().next_station)))


def label_long_delay_stations(df, percentile=90):
    """
    Label long delay stations

    :param df: a pandas dataframe with OTP data
    :param percentile: percentile value to use for setting the threshold. Default: 90

    :return a dataframe where the long delay stations are labeled
    """

    df = df.copy()
    
    station_delay = calculate_station_delay(df=df)
    delay_threshold = calculate_delay_threshold(df=station_delay, percentile=percentile)
    long_delay_station = calculate_long_delay_stations(station_delay=station_delay, delay_threshold=delay_threshold)

    # label long duration stations
    long_delay_station['long_delay_station'] = True
    df = df.merge(long_delay_station.reset_index()[['direction', 'next_station', 'long_delay_station']],
                  on=['direction', 'next_station'], how='left')
    df['long_delay_station'] = df.long_delay_station.fillna(False)

    return df


def calculate_longest_delays_over_past_week(df):
    """
    Calculate longest delays in the past week

    :param df. a pandas dataframe with OTP data

    :return a pandas dataframe with longest delays and longest delays in teh past week
    """
    
    # Warning, it takes a few minutes to calculate
    df = df.copy()

    # sort values according to date
    df = df.sort_values('date')

    # loop through each date and get average delay for past 7 days per train_id and station
    last_week_delay = []
    for d in df.date.unique():
        # define threshold
        last_d = d - np.timedelta64(7, 'D')
        
        # calculate delay
        last_week_delay0 = df.query("date >= '%s' & date < '%s'" % (last_d, d))\
            .merge(df.query("date == '%s'" % d)[['train_id', 'next_station']],
                   on=['train_id', 'next_station']).\
            groupby(['train_id', 'next_station']).delay.mean().reset_index(name='last_week_delay')
        
        # add date and collect
        last_week_delay0['date'] = d
        last_week_delay.append(last_week_delay0)
        
    last_week_delay = pd.concat(last_week_delay)

    # merge with df_otp
    df = df.merge(last_week_delay, on=['train_id', 'next_station', 'date'], how='left')

    # set missing to 0 (no available last 7 days date)
    df['last_week_delay'] = df.last_week_delay.fillna(0)

    return df, last_week_delay


def get_track_changes(df):
    """
    Calculate track changes

    :param df. a pandas dataframe with train_view data

    :returns a pandas dataframe with track changes
    """

    df = df.copy()
    # convert track changes to binary
    df['track_change_formatted'] = df.track_change.astype(str)
    df['track_change_formatted'] = df.track_change.apply(lambda x: 0 if x == '-1' else 1)

    # for each train_id, station & date, check whether there was a track change and how much was the total delay
    df_track_changes = df.groupby(['train_id',
                                   'next_station',
                                   'date']
                                  ).agg({'track_change_formatted': lambda x: 1 if sum(x) else 0, 'delay': 'sum'})

    return df_track_changes


def calculate_track_changes_frequency(df):
    """
    Calculate track change frequency per station and date

    :param df. a pandas dataframe with track changes

    :return a pandas dataframe with track change frequency per station and date
    """

    df = df.copy()

    # calculate track change frequency per station and date
    track_change_frequency_date = df.reset_index(level='train_id').\
        groupby([pd.Grouper(level='next_station'), pd.Grouper(level='date')]).\
        apply(lambda x: sum(x['track_change_formatted']) / float(len(x))).reset_index(name='track_change_frequency')

    return track_change_frequency_date


def calculate_aggregated_track_changes(df):
    """
    Calculate average track change frequency per station (across dates)

    :param df. a pandas dataframe with track change frequency date

    :return a pandas dataframe with average track change frequency per station
    """

    df = df.copy()
    
    # calculate average track change frequency per station (across dates)
    track_change_frequency = df.groupby('next_station').track_change_frequency.\
        mean().reset_index(name='track_change_frequency').sort_values('track_change_frequency', ascending=False)

    return track_change_frequency
