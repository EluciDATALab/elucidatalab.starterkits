# ©, 2022, Sirris
# owner: HCAB

import pkg_resources
import os

import pandas as pd

from starterkits import pipeline, DATA_PATH
import zipfile

reordered_nodes = ['Fremont_Bridge', 'Spokane_St', '2nd_Ave', 'NW_58th_St', '39th_Ave', '26th_Ave']
reordered_columns = [f'Total {col}' for col in reordered_nodes]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def try_package(x):
    pkg, pkg_version = x.split('==')
    try:
        version = pkg_resources.get_distribution(pkg).version
        if pkg_version == version:
            pass
        else:
            print((f'Warning: this notebook was tested with version {pkg_version} of {pkg},'
                   f'but you have {version} installed'))
    except Exception as e:
        print(e)


def read_data(name):
    if not os.path.exists(os.path.join('data', name + '.csv')):
        pipeline.download_from_repo(os.path.join('data', name + '.csv'),
                                    'data/StarterKits/D3.1/' + name + '.csv').make()
    data = pd.read_csv(os.path.join('data', name + '.csv'), header=0,
                       names=['Date', 'Total', 'Direction 0', 'Direction 1'])
    data['Node'] = name
    data['Timestamp'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    return data


def read_fremont_data(name):
    if not os.path.exists(os.path.join('data', name + '.csv')):
        pipeline.download_from_repo(os.path.join('data', name + '.csv'),
                                    'data/StarterKits/D3.1/' + name + '.csv').make()
    data = pd.read_csv(os.path.join('data', name + '.csv'), header=0, names=['Date', 'Direction 1', 'Direction 0'])
    data['Node'] = name
    data['Timestamp'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    data['Total'] = data['Direction 0'] + data['Direction 1']
    return data


def read_all_data():
    df1 = read_data('NW_58th_St')
    df2 = read_data('39th_Ave')
    df3 = read_data('Spokane_St')
    df4 = read_data('26th_Ave')
    df5 = read_data('2nd_Ave')
    df6 = read_fremont_data('Fremont_Bridge')
    data = pd.concat([df1, df2, df3, df4, df5, df6], sort=False)
    data.fillna(0, inplace=True)

    data['Date'] = pd.to_datetime(data['Timestamp'].dt.date)
    data['Time'] = data['Timestamp'].dt.time
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek + 1

    data = data[['Node', 'Timestamp', 'Date', 'Time', 'Hour', 'DayOfWeek', 'Direction 0', 'Direction 1', 'Total']]

    return data


def get_data(force=False):
    """Extract dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally.
    returns: Seattle bike data
    """ 
    def extract_csv_file(file_name):
        target_folder = DATA_PATH / 'SK_3_1'
        zip_file_name = pipeline.download_from_repo(
            'SK_3_1',
            'Seattle_bike_data.csv.zip',
            str(target_folder / 'Seattle_bike_data.csv.zip'),
            force=force)

        with zipfile.ZipFile(zip_file_name, 'r') as zipped_file:
            zipped_file.extract('Seattle_bike_data.csv', path=str(target_folder))
        os.remove(zip_file_name)
        return file_name
    f_name = (pipeline.File(extract_csv_file,
                            str(DATA_PATH / 'SK_3_1' / 'Seattle_bike_data.csv'))
              .make(force=force))

    df = pd.read_csv(f_name, sep=',', index_col=0, parse_dates=['Timestamp', 'Date'])
    df['Time'] = df['Timestamp'].dt.time

    return df
