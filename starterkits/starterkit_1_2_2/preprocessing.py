# ©, 2024, Sirris
# owner: FFNG

from starterkits.preprocessing.normalization import normalize_along_rows
from starterkits.starterkit_1_2_2.nmf_profiling import derive_df_orders, derive_df_vib
from starterkits.starterkit_1_2_2.data_handler import fetch_and_unzip_data, load_data
from starterkits.starterkit_1_2_2 import BASE_PATH_HEALTHY, BASE_PATHS_TEST, PITTING_LEVELS
from starterkits.starterkit_1_2_2 import SETUP
import random
import os
import pandas as pd
import pickle
from tqdm import tqdm
import glob


def _convert_to_frequency_domain(nperseg=10240, nfft=None, fs=20480, split=0.75):

    noverlap = nperseg // 2
    file_names_healthy = glob.glob(os.path.join(BASE_PATH_HEALTHY, '*.txt'))
    data_healthy, f = load_data(file_names_healthy, nperseg=nperseg, noverlap=noverlap, nfft=nfft, fs=fs)

    split_id = int(len(data_healthy) * split)
    random.Random(0).shuffle(data_healthy)
    data_healthy_train = data_healthy[:split_id]
    data_healthy_test = data_healthy[split_id:]

    return data_healthy_train, data_healthy_test, f


def _transform_orders_and_bin(data_healthy_train, f):

    df_vib_train = derive_df_vib(data_healthy_train, f)
    df_orders_train, meta_data_train = derive_df_orders(df_vib_train, SETUP, f, verbose=False)
    df_orders_train[meta_data_train.columns] = meta_data_train

    return df_orders_train, meta_data_train


def _normalize_orders(df_orders_train):

    cols = df_orders_train.columns
    BAND_COLS = cols[cols.str.contains('band')].tolist()
    df_V_train = normalize_along_rows(df_orders_train, BAND_COLS)

    return df_V_train


def get_and_preprocess_healthy_data():
    # fetch_and_unzip_data()
    data_healthy_train, df_data_healthy_test, f = _convert_to_frequency_domain()
    df_orders_train, meta_data_train = _transform_orders_and_bin(data_healthy_train, f)
    df_V_train = _normalize_orders(df_orders_train)

    return df_V_train, meta_data_train, df_data_healthy_test, f


def get_and_preprocess_unhealthy_data(df_data_healthy_test, f):

    # convert pitting test samples to orders
    df_orders_test_pitting_dict = {}
    meta_data_test_pitting_dict = {}

    for lvl, path in tqdm(list(zip(PITTING_LEVELS, BASE_PATHS_TEST)),
                          desc='Extracting and order-transforming test data'):
        # load data for each level of pitting
        fnames = glob.glob(os.path.join(path, '*.txt'))
        nperseg = 10240
        noverlap = nperseg // 2
        nfft = None
        fs = 20480
        data_test, f = load_data(fnames, nperseg=nperseg, noverlap=noverlap, nfft=nfft, fs=fs, base_path=path,
                                 use_train_data_for_validation=True)

        # extract vibration data
        df_vib_test_unhealthy = derive_df_vib(data_test, f)

        # convert to orders and derive meta data

        df_orders_test_pitting_, meta_data_test_pitting_ = derive_df_orders(df_vib_test_unhealthy, SETUP, f,
                                                                            verbose=False)
        rpm = meta_data_test_pitting_['rotational speed [RPM]']
        torque = meta_data_test_pitting_['torque [Nm]']
        run = meta_data_test_pitting_['sample_id']
        meta_data_test_pitting_['unique_sample_id'] = rpm.astype(str) + '_' + torque.astype(str) + '_' + run.astype(
            str) + f'_pitting_level_{lvl}'
        df_orders_test_pitting_['unique_sample_id'] = meta_data_test_pitting_['unique_sample_id']
        df_orders_test_pitting_dict[lvl] = df_orders_test_pitting_
        meta_data_test_pitting_dict[lvl] = meta_data_test_pitting_

    # convert healthy test samples to orders
    df_vib_test_healthy = derive_df_vib(df_data_healthy_test, f)
    df_orders_test_healthy, meta_data_test_healthy = derive_df_orders(df_vib_test_healthy, SETUP, f, verbose=False)
    meta_data_test_healthy['unique_sample_id'] = meta_data_test_healthy['unique_sample_id'] + '_healthy'
    df_orders_test_healthy['unique_sample_id'] = meta_data_test_healthy['unique_sample_id']

    # concat all pitting levels samples
    df_orders_test_pitting = pd.concat(list(df_orders_test_pitting_dict.values()))
    meta_data_test_pitting = pd.concat(list(meta_data_test_pitting_dict.values()))

    # merge healthy and unhealthy samples
    # only use operating modes in the test set that are also in the training set
    om_test_healthy = (meta_data_test_healthy['rotational speed [RPM]'].astype(str) + '_' +
                       meta_data_test_healthy['torque [Nm]'].astype(str))
    om_test_pitting = (meta_data_test_pitting['rotational speed [RPM]'].astype(str) + '_' +
                       meta_data_test_pitting['torque [Nm]'].astype(str))
    new_meta_data_test_pitting_without_missing_oms = meta_data_test_pitting[
        om_test_pitting.isin(om_test_healthy)]
    new_df_orders_test_pitting_without_missing_oms = df_orders_test_pitting[
        om_test_pitting.isin(om_test_healthy)]

    # sample equal amount of samples from healthy and faulty data
    om_test_healthy_with_run = (meta_data_test_healthy['rotational speed [RPM]'].astype(str) + '_'
                                + meta_data_test_healthy['torque [Nm]'].astype(str) + '_'
                                + meta_data_test_healthy['sample_id'].astype(str))
    n_samples = len(om_test_healthy_with_run.unique())
    samples = new_df_orders_test_pitting_without_missing_oms['unique_sample_id'].sample(n_samples,
                                                                                        random_state=0,
                                                                                        replace=False)
    new_meta_data_test_pitting = new_meta_data_test_pitting_without_missing_oms[
        new_meta_data_test_pitting_without_missing_oms['unique_sample_id'].isin(samples)]
    new_df_orders_test_pitting = new_df_orders_test_pitting_without_missing_oms[
        new_df_orders_test_pitting_without_missing_oms['unique_sample_id'].isin(samples)]
    df_orders_test = pd.concat([df_orders_test_healthy, new_df_orders_test_pitting]).reset_index(drop=True)
    meta_data_test = pd.concat([meta_data_test_healthy, new_meta_data_test_pitting]).reset_index(drop=True)

    return df_orders_test, meta_data_test
