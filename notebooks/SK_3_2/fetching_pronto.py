from elucidata import resources
import zipfile
import pandas as pd
import glob
from elucidata.resources import pipeline
import os
from joblib import Memory

root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def path(x=None):
    """Construct a relative path."""
    global root_dir
    if x is None:
        return root_dir
    return os.path.join(root_dir, x)


memory = Memory(path('cache'))


def cache():
    return memory.cache()


def clear_cache():
    memory.clear()


class DataFetcher(object):
    """
        class to access the data of PICESC from artemis.
        Available datasets are june, august and october 2021

        Usage:
            import picesc
            df_data = picesc.fetching_data.DataFetcher().get_all_data() # to access all csv files

            To load the original dataset from June, use:
            df_data = picesc.fetching_data.DataFetcher('june-2021').get_all_data()

    """

    def __init__(self, data_version=None):
        self.list_csv = ['2015_station_data',"2015_trip_data","2015_weather_data"]

    def _unzip_data_file(self, path_files, path_zips):
        zip_path_artemis = 'data/StarterKits/D3.2/SK-3-2-Pronto.zip'
        zip_path = pipeline.DownloadFile(local_path=path(path_zips),
                                         remote_path=zip_path_artemis).make()
        target_main_dir = path(path_files)
        target_dir = path(path_files + '/Pronto-data')
        if not os.path.exists(target_dir) or len(os.listdir(target_dir)) == 0:
            print('Unzipping in', target_dir)
            with zipfile.ZipFile(zip_path, mode='r') as zip_ref:
                zip_ref.extractall(target_main_dir)


    def _get_one_csv(self, name, path_files, path_zips, path_unzips, **csv_kwargs):
        def get_data():
            if not os.path.exists(path_unzips):
                self._unzip_data_file(path_files, path_zips)
            path_csv = glob.glob(path(path_unzips + "/" + name + "*.csv"))[0]
            df_csv = pd.read_csv(path(path_csv), sep=",")
            return df_csv

        return (resources.pipeline.DataFrameFile(
            generate_df_func=get_data,
            local_path=path('cache/'  + str(name) + '.csv'),
            remote_path='outputs/Pronto/' + str(name) + '.csv'))

    def get_all_data(self, names=None, force=False):
        """
                Function to extract all data

                it returns a dictionary containing all or the selected dataframes.

                usage:
                    import SK3-2
                    SK3-2.fetching_pronto.DataFetcher().get_all_data()
            param: names the list of dataframes needed.
                By defaults, it returns all of them (['2015_station_data',"2015_trip_data","2015_weather_data"])
            return: a dictionary containing the requested dataframes

        """
        path_zips = 'cache/zip/SK-3-2-Pronto.zip'
        path_files = 'cache/raw-unzipped/'
        path_unzips = 'cache/raw-unzipped/SK-3-2-Pronto'
        if names is None:
            names = self.list_csv

        res = {}
        for name in names:
            tmp_df = self._get_one_csv(name, path_files, path_zips, path_unzips).make(force=force)
            if name == '2015_station_data':
                tmp_df = tmp_df.rename({"long":"lon"}, axis=1)
            if name == '2015_trip_data':
                tmp_df [['starttime','stoptime']] = tmp_df [['starttime','stoptime']].apply(pd.to_datetime)
            res[name.split('_')[1].lower()] = tmp_df
        return res

    def get_stations(self, force=False):
        """
                Function to extract the data about the stations (as a dataframes).

                usage:
                    import SK3-2
                    SK3-2.fetching_pronto.DataFetcher().get_stations()
            param: force: to force the reload of the data from artemis
            return: the requested dataframes

        """
        return self.get_all_data(names=['2015_station_data'], force=force)['station']

    def get_trips(self, force=False):
        """
                Function to extract the data about the trips (as a dataframes).

                usage:
                    import SK3-2
                    SK3-2.fetching_pronto.DataFetcher().get_trips()
            param: force: to force the reload of the data from artemis
            return: the requested dataframes

        """
        return self.get_all_data(names=['2015_trip_data'], force=force)['trip']

    def get_weather(self, force=False):
        """
                Function to extract the data about the weather (as a dataframes).

                usage:
                    import SK3-2
                    SK3-2.fetching_pronto.DataFetcher().get_weather()
            param: force: to force the reload of the data from artemis
            return: the requested dataframes

        """
        return self.get_all_data(names=['2015_weather_data'], force=force)['weather']

    def get_merged_without_elevation(self, force=False):
        """
                Function to extract the trips and station data merged (without elevation)

                usage:
                    import SK3-2
                    SK3-2.fetching_pronto.DataFetcher().get_merged_without_elevation()
            param: force: to force the reload of the data from artemis
            return: the requested dataframe

        """
        df_trips = self.get_trips()
        df_stations = self.get_stations()
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

    def get_merged_data(self, force=False):
        """
                Function to extract the trips and station data merged (without elevation)

                usage:
                    import SK3-2
                    SK3-2.fetching_pronto.DataFetcher().get_merged_without_elevation()
            param: force: to force the reload of the data from artemis
            return: the requested dataframe

        """
        df_trips = self.get_trips()
        df_stations = self.get_stations()
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

