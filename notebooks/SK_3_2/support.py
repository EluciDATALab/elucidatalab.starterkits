# ©, 2022, Sirris
# owner: AEGH

import datetime as dt
import pandas as pd
import numpy as np
from starterkits import pipeline, DATA_PATH
import zipfile

def extract_destination_station_frequency_per_gender(df_trips):
    """
    Extract per gender (female and male) how frequently each destination
    station was reached for trips.
    
    :param df_trips: A pandas dataframe wit trips data

    :returns: A pandas dataframe with frequency and ratios per gender per station
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

def read_data(name, force=False):
    """
    Import CSV files

    :param name: Name of the file
    :param force: to force the reload of the data from artemis

    :returns: read pronto csv files
    """

    get_data(force=force)
    data = pd.read_csv(str(DATA_PATH / 'SK_3_2'  / f'{name}.csv'))
    
    return data

def preprocess_station_data(df):
    """
    Preprocess station data

    :param df: A pandas dataframe with station data

    :returns: Preprocessed station data
    """
    df = df.copy()
    df = df.rename({"long":"lon"}, axis=1)

    return df

def get_trips(force=False):
    """
    Get trip data

    :param force: to force the reload of the data from artemis

    :returns: trip data
    """
    df= read_data(name='2015_trip_data', force=force)
    df[['starttime','stoptime']] = df [['starttime','stoptime']].apply(pd.to_datetime)

    return df 

def get_stations(force=False):
    """
    Get station data

    :param force: to force the reload of the data from artemis

    :returns: station data
    """
    df= read_data('2015_station_dataV2', force=force)
    df = preprocess_station_data(df=df)

    return df 

def get_merged_without_elevation(force=False):
    """
    Function to extract the trips and station data merged (without elevation)

    :param force: to force the reload of the data from artemis

    :returns: trips and station merged data without elevation
    """
    df_trips = get_trips(force=force)
    df_stations = get_stations(force=force)
    df_trips = (df_trips
                # from
                .merge(df_stations[['terminal', 'lat', 'lon']], left_on=['from_station_id'], right_on=['terminal'])
                .drop(['terminal'], axis=1)
                .rename(columns={'lat': 'from_station_lat', 'lon': 'from_station_lon'})
                # to
                .merge(df_stations[['terminal', 'lat', 'lon']], left_on=['to_station_id'], right_on=['terminal'])
                .drop(['terminal'], axis=1)
                .rename(columns={'lat': 'to_station_lat', 'lon': 'to_station_lon'}))

    return df_trips

def get_merged_data_with_elevation(force=False):
    """
    Function to extract the trips and station data merged (with elevation)

    :param force: to force the reload of the data from artemis

    :returns: trips and station merged data with elevation
    """
    df_trips = get_trips(force=force)
    df_stations = get_stations(force=force)
    df_trips = (df_trips
        # from
        .merge(df_stations[['terminal', 'lat', 'lon', 'elevation']], left_on=['from_station_id'], right_on=['terminal'])
        .drop(['terminal'], axis=1)
        .rename(columns={'lat': 'from_station_lat', 'lon': 'from_station_lon', 'elevation': 'from_station_elevation'})
        # to
        .merge(df_stations[['terminal', 'lat', 'lon', 'elevation']], left_on=['to_station_id'], right_on=['terminal'])
        .drop(['terminal'], axis=1)
        .rename(columns={'lat': 'to_station_lat', 'lon': 'to_station_lon', 'elevation': 'to_station_elevation'}))

    return df_trips

def extract_csv_file(zip_fname):
    """
    Extract CSV from zip file

    :param filename: Name of the file
    :param force: to force the reload of the data from artemis

    :returns: return name of the file
    """
    
    archive = zipfile.ZipFile(zip_fname)
    for file in archive.namelist():
        archive.extract(file, str(DATA_PATH) + '/SK_3_2/')


def get_data(force=False):
    """
    Extract data zip file from server and unzip it. This dataset will be saved locally.

    :param force: to force the reload of the data from artemis
    """ 

    zip_fname = pipeline.download_from_repo(starterkit='SK_3_2',
                                dataset = 'SK-3-2-Pronto.zip',
                                fname = str(DATA_PATH / 'SK_3_2' / 'SK-3-2-Pronto.zip'),
                                force=force)

    extract_csv_file(zip_fname=zip_fname)


def get_weekdays():
    """
    Get the days of the week

    :returns: a dictionary with weekday and weekend
    """
    week_days = {'weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], 
             'weekend': ['Sat', 'Sun']}
    
    return week_days


def get_trips_weekend(df_trips):
    """
    Get trip data of the weekend

    :param df_trips: a pandas dataframe with trip data

    :returns: trip data on the weekend
    """
    week_days = get_weekdays()
    df_trips_weekend = df_trips[df_trips.day.isin(week_days['weekend'])]

    return df_trips_weekend

def preprocess_trips_dataset(df_trips):
    """
    Preprocess trips dataset

    :param df_trips: a pandas dataframe with trip data

    :returns: preprocessed trip data
    """
    df_trips['tripdurationTimeDelta'] = df_trips['tripduration'].map(lambda x: dt.timedelta(seconds=x))
    df_trips['tripdurationMinutes'] = df_trips['tripduration'] / 60

    df_trips['birthyear'] = df_trips['birthyear'].astype(int, errors='ignore')
    df_trips['age'] = (df_trips.starttime.dt.year - df_trips.birthyear).astype(int, errors='ignore')
    df_trips['month'] = pd.Categorical(df_trips.starttime.apply(lambda x: dt.datetime.strftime(x, '%b')),
                                       ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    df_trips['day'] = pd.Categorical(df_trips.starttime.apply(lambda x: dt.datetime.strftime(x, '%a')),
                                     ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    df_trips['hour'] = df_trips.starttime.dt.hour

    df_trips.set_index(df_trips['starttime'], inplace=True)
    df_trips.index.name = None
    
    return df_trips


def get_trips_top_arrival_morning_rush(df_trips, n=10):
    """
    Get trip data during morning rush

    :param df_trips: a pandas dataframe with trip data
    :param n: Count of datapoints to return 

    :returns: top n trip data points during morning rush
    """
    week_days=get_weekdays()
    df_trips_weekdays = df_trips[df_trips.day.isin(week_days['weekday'])]
    morning_rush_hour = df_trips_weekdays.hour.isin([7, 8, 9])
    df_trips_topArrival_morning_rush = (df_trips_weekdays[morning_rush_hour]
        .groupby(['to_station_name'])
        .agg({'to_station_id': 'count', 'to_station_lat': 'first', 'to_station_lon': 'first'})
        .sort_values('to_station_id', ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={'to_station_id': 'count'}))
    
    return df_trips_topArrival_morning_rush
    
def get_trips_top_arrival_evening_rush(df_trips, count=10):
    """
    Get trip data during evening rush

    :param df_trips: a pandas dataframe with trip data
    :param n: Count of datapoints to return 

    :returns: top n trip data points during evening rush
    """
    week_days=get_weekdays()
    df_trips_weekdays = df_trips[df_trips.day.isin(week_days['weekday'])]
    evening_rush_hour = df_trips_weekdays.hour.isin([16, 17, 18])
    df_trips_topArrival_evening_rush = (df_trips_weekdays[evening_rush_hour]
        .groupby(['to_station_name'])
        .agg({'to_station_id' : 'count', 'to_station_lat': 'first', 'to_station_lon': 'first'})
        .sort_values('to_station_id', ascending=False)
        .head(count)
        .reset_index()
        .rename(columns={'to_station_id': 'count'}))
    
    return df_trips_topArrival_evening_rush

def get_trip_duration_gender(df_trips):
    """
    Get trip data per gender (man/women)

    :param df_trips: a pandas dataframe with trip data

    return trip data information per gender (man/women)
    """
    trip_duration_gender = (df_trips
    .groupby('gender')
    .agg({'trip_id': 'count', 
          'tripdurationMinutes': 'mean'})
    .rename(columns={'trip_id': 'Number of Trips',
                    'tripdurationMinutes':'Trip Duration in Minutes'}))
    
    return trip_duration_gender

def get_data_stats(df):
    """
    Get basic stats of the trips dataset

    :param df: A pandas dataframe with trips data
    """

    df = df.copy()

    nof_trips = df.index.size
    start_date = df.starttime.min().date()
    end_date = df.stoptime.max().date()
    print(f"Dataset contains {nof_trips} trips from {start_date} to {end_date}.")
    print(f"Number of bikes registered in the Pronto system: {df.bikeid.nunique()}")
    print(f"Number of stations: {df.from_station_id.nunique()}")

def get_trips_from_statios(df_trips):
    """
    Get informaion about the source station

    :param df_trips: A pandas dataframe with trips data

    :returns: 
    """
    df_trips_from_stations = (df_trips
    .groupby('from_station_name')
    .agg({'from_station_id': 'count',
          'from_station_lat': 'first',
          'from_station_lon': 'first',
          'from_station_elevation':'first'})
    .reset_index()
    .rename(columns={'from_station_name': 'station_name',
                     'from_station_id': 'from_station',
                     'from_station_lat': 'lat',
                     'from_station_lon': 'lon',
                     'from_station_elevation':'elevation'}))
    
    return df_trips_from_stations

def get_trips_to_stations(df_trips):
    """
    Get informaion about the destination station

    :param df_trips: A pandas dataframe with trips data

    :returns: 
    """
    df_trips_to_stations = (df_trips
    .groupby('to_station_name')
    .size()
    .reset_index(name='to_station')
    .rename(columns={'to_station_name': 'station_name'}))
    
    return df_trips_to_stations

def get_trips_per_stations(df_trips):
    """
    Get trips information per station 

    :param df_trips: A pandas dataframe with trips and station data

    :returns: a pandas dataframe with the trips information per station
    """
    df_trips_from_stations=get_trips_from_statios(df_trips)
    df_trips_to_stations=get_trips_to_stations(df_trips)
    df_trips_stations = df_trips_to_stations.merge(df_trips_from_stations, on='station_name')
    
    return df_trips_stations

def get_unbalanced_stations(df, n_stations=10):
    """
    Get data for the unbalanced station

    :param df_trips: A pandas dataframe with trips and station data

    :returns: A pandas dataframe with data about unbalanced stations
    """
    df_trips_stations=get_trips_per_stations(df)
    df_trips_stations['TripDifference'] = df_trips_stations.to_station - df_trips_stations.from_station 
    df_trips_stations['AbsTripDifference'] = np.abs(df_trips_stations['TripDifference'])
    df_topUnbalanced_stations = df_trips_stations.sort_values('AbsTripDifference', ascending=False)[:n_stations]

    return df_topUnbalanced_stations


def get_gender_count(df_trips):
    """
    Get gender count

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with gender count
    """
    gender_count = (df_trips[df_trips.gender.isin(['Male', 'Female', 'Other'])]
    .groupby('gender')
    .size()
    .reset_index(name='count'))

    return gender_count


def get_stations_ordered_by_count(df_trips):
    """
    Get stations data ordered by count

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with station data ordered by count
    """
    stations_ordered_by_count = (df_trips
    .groupby('to_station_name')
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False))

    return stations_ordered_by_count


def get_trip_hour(df_trips, days):
    """
    Get stations data ordered by count

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with station data ordered by count
    """
    df_trip_hour = (df_trips[df_trips.day.isin(days)]
            .groupby('hour')
            .size()
            .reset_index(name='ct'))
    df_trip_hour.ct = df_trip_hour.ct / len(days)  # normalize by number of days

    return df_trip_hour

def get_trips_dow(df_trips):
    """
    Get days of the week for the trip data

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with trips data and days of the week
    """
    df_trips_dow = (df_trips
        .groupby('day')
        .agg({'tripdurationMinutes': 'median', 'starttime': 'count'})
        .reset_index()
        .rename(columns={'starttime': 'n_trips'}))
    
    return df_trips_dow

def get_trips_moy(df_trips):
    """
    Get month of the year for the trip data

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with trips data and month of the year
    """
    df_trips_moy = (df_trips
        .groupby('month')
        .agg({'tripdurationMinutes': 'median', 'starttime': 'count'})
        .reset_index()
        .rename(columns={'starttime': 'n_trips'}))
    
    return df_trips_moy


def get_trips_duration(df_trips):
    """
    Get trips duration

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with trips duration
    """
    df_trip_duration = (df_trips
    .groupby(['to_station_name', 'from_station_name'])
    .tripdurationMinutes
    .agg(['median', 'count'])
    .reset_index()
    .rename(columns={'to_station_name': 'to_station', 
                     'from_station_name': 'from_station',
                     'median': 'median duration (min)'}))

    return df_trip_duration


def get_unbalanced_trip_duration(df, n_stations=10):
    """
    Get trips duration for the unbalanced stations

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with trips duration of the unbalanced stations
    """
    
    df_trip_duration=get_trips_duration(df)
    df_trip_duration = df_trip_duration[df_trip_duration.to_station != df_trip_duration.from_station]
    
    df_topUnbalanced_stations=get_unbalanced_stations(df, n_stations=n_stations)
    stations_unbalanced = df_topUnbalanced_stations.station_name
    df_trip_duration = df_trip_duration[(df_trip_duration['to_station'].isin(stations_unbalanced)) &
                                        (df_trip_duration['from_station'].isin(stations_unbalanced))]
    
    return df_trip_duration

def get_arrival_morning_rush_hour(df):
    """
    Get morning rush hours for arrival trips

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with morning rush hours for arrival trips
    """

    week_days = get_weekdays()
    df_trips_topArrival_morning_rush = get_trips_top_arrival_morning_rush(df)
    df_tripsTo_topStationArrival_morning_rush = (
        df[df['to_station_name'].isin(df_trips_topArrival_morning_rush.to_station_name)])

    df_tripsTo_topStationArrival_morningRushHour = (
        df_tripsTo_topStationArrival_morning_rush[(df_tripsTo_topStationArrival_morning_rush
                                                   .day
                                                   .isin(week_days['weekday']))])
    return df_tripsTo_topStationArrival_morningRushHour


def get_arrival_station_frequency_wm_morning_rush_hour(df):
    """
    Get the frequency of stations during morning rush hours for arrival trips

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with morning rush hours for arrival trips
    """
    
    df_tripsTo_topStationArrival_morningRushHour=get_arrival_morning_rush_hour(df=df)
    df_arrivalStationFrequency_wm_morningRushHour = (
        extract_destination_station_frequency_per_gender(df_tripsTo_topStationArrival_morningRushHour).reset_index())
    
    return df_arrivalStationFrequency_wm_morningRushHour



def get_arrival_station_frequency_wm_weekend_women(df, n_stations=10):
    """ 
    Get the frequency of arrival stations during weekend for women

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with the frequency of arrival stations during weekend for women
    """
    
    df_trips_weekend=get_trips_weekend(df)
    df_arrivalStationFrequency_wm_weekend = extract_destination_station_frequency_per_gender(df_trips_weekend)
    df_arrivalStationFrequency_wm_weekend_women = (
        df_arrivalStationFrequency_wm_weekend.sort_values('ratio_f2m', ascending=False)
                                         .head(n_stations)
                                         .reset_index())
    return df_arrivalStationFrequency_wm_weekend_women

def get_arrival_station_frequency_wm_weekend_men(df, n_stations=10):
    """ 
    Get the frequency of arrival stations during weekend for men

    :param df_trips: A pandas dataframe with trips data

    :returns: A pandas dataframe with the frequency of arrival stations during weekend for men
    """
    
    df_trips_weekend=get_trips_weekend(df)
    df_arrivalStationFrequency_wm_weekend = extract_destination_station_frequency_per_gender(df_trips_weekend)
    df_arrivalStationFrequency_wm_weekend_men = (
        df_arrivalStationFrequency_wm_weekend.sort_values('ratio_m2f', ascending=False)
                                         .head(n_stations)
                                         .reset_index())
    return df_arrivalStationFrequency_wm_weekend_men


