# ©, 2022, Sirris
# owner: HCAB

import os
import pandas as pd

from starterkits import pipeline, DATA_PATH
import zipfile


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
