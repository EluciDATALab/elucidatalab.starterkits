# ©, 2024, Sirris
# owner: FFNG

import zipfile
import os
import pandas as pd
from scipy.signal import stft, welch
from pathlib import Path, PosixPath
import requests
from tqdm import tqdm
from starterkits.pipeline import download_from_url
from starterkits.starterkit_1_2_2 import DATA_PATH, FNAME, BASE_PATH_HEALTHY


def fetch_and_unzip_data(force=False):
    """ Fetch and unzip data from the remote server. """
    url = 'https://phm-datasets.s3.amazonaws.com/Data_Challenge_PHM2023_training_data.zip'

    fname_zip = f'{FNAME}.zip'

    local_path_zipped = os.path.join(DATA_PATH, fname_zip)
    local_path_unzipped = os.path.join(DATA_PATH, FNAME)

    download_from_url(url, local_path_zipped, force=force)

    if not os.path.exists(local_path_unzipped) or force:
        assert os.path.exists(local_path_zipped), f'Could not find {local_path_zipped}'
        print('Unzipping data...')
        with zipfile.ZipFile(local_path_zipped, 'r') as zip_ref:
            zip_ref.extractall(DATA_PATH)

    return local_path_unzipped


def load_train_data(rpm, torque, run, base_path=BASE_PATH_HEALTHY):
    """ Load training data for a given run, rpm and torque. """
    path = os.path.join(base_path, f'V{rpm}_{torque}N_{run}.txt')
    df = pd.read_csv(path, names=['x', 'y', 'z', 'tachometer'], delimiter=' ')
    return df


def load_test_data(rpm, torque, run, base_path=BASE_PATH_HEALTHY):
    """ Load test data for a given run, rpm and torque. """
    path = os.path.join(base_path, f'{run}_V{rpm}_{torque}N.txt')
    df = pd.read_csv(path, names=['x', 'y', 'z', 'tachometer'], delimiter=' ')
    return df


def extract_process_parameters(file_path, use_train_data_for_validation=True):
    parts = file_path.split('/')
    filename = parts[-1]  # Extract the filename from the path
    if use_train_data_for_validation:
        v_value, n_value, sample_number = filename.split('_')  # Extract V, N, and sample number
        return int(v_value[1:]), int(n_value[:-1]), int(sample_number.split('.')[0])
    else:
        sample_number, v_value, n_value = filename.split('_')
        return int(v_value[1:]), int(n_value.split('.')[0][:-1]), int(sample_number)


def load_data(fnames, use_train_data_for_validation=True, base_path=BASE_PATH_HEALTHY, **kwargs):
    """ Load the complete data set.

    :param fnames: List of strings. Contains all the file names which should be loaded,
        e.g., ['V100_50N_1.txt', 'V3600_100N_4.txt', ...].
    :param use_train_data_for_validation: _description_, defaults to True.
    :param base_path: _description_, defaults to BASE_PATH_HEALTHY
    :return: _description_
    """
    data = []
    f = None
    for fn in fnames:
        rpm, torque, run = extract_process_parameters(fn, use_train_data_for_validation=use_train_data_for_validation)
        if use_train_data_for_validation:
            df = load_train_data(rpm, torque, run, base_path=base_path)
        else:
            df = load_test_data(rpm, torque, run, base_path=base_path)

        f, t, stft_x = stft(df['x'], **kwargs)
        f, t, stft_y = stft(df['y'], **kwargs)
        f, t, stft_z = stft(df['z'], **kwargs)
        f, psd_x = welch(df['x'], **kwargs)
        f, psd_y = welch(df['y'], **kwargs)
        f, psd_z = welch(df['z'], **kwargs)
        data.append({
            'rpm': rpm,
            'torque': torque,
            'sample_id': run,
            'unique_sample_id': f'{rpm}_{torque}_{run}',  # Remove the '.txt' extension and convert to integer
            'vibration_time_domain': df,
            'stft_x': stft_x,
            'stft_y': stft_y,
            'stft_z': stft_z,  # Remove the '.txt' extension and convert to integer
            'psd_x': psd_x,
            'psd_y': psd_y,
            'psd_z': psd_z
        })
    return data, f
