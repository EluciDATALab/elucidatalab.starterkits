"""
Code related to profiling batches via non-negative matrix factorization (NMF).
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from starterkits.utils import distance_metrics
from starterkits.preprocessing.normalization import normalize_along_rows


def extract_nmf_incremental(df_V, max_n_components, timestamps=None, verbose=True):

    V = df_V.to_numpy()
    range_components = range(1, max_n_components+1)[::-1]
    tqdm_description = 'Fitting NMF with varying number of components'
    tqdm_range_components = tqdm(range_components, desc=tqdm_description, disable=(not verbose))

    list_models = []
    previous_W = None
    previous_H = None
    for _n_components in tqdm_range_components:
        if previous_W is None or previous_H is None:
            # For the first (largest) decomposition, use the default 'nndsvd' initialization
            nmf = NMF(n_components=_n_components, init='nndsvd', max_iter=1000, random_state=42)
            W = nmf.fit_transform(V)
            H = nmf.components_
        else:
            # For subsequent decompositions, use the results of the previous decomposition for warm start
            nmf = NMF(n_components=_n_components, init='custom', max_iter=1000, random_state=42)
            W = previous_W[:, :_n_components].copy(order='C')  # Use only the first n_components columns
            H = previous_H[:_n_components, :].copy(order='C')  # Use only the first n_components rows

            W = nmf.fit_transform(V, W=W, H=H)
            H = nmf.components_
        V_reconstructed = nmf.inverse_transform(W)

        model_dict = {
            # number of components
            'n_components': _n_components,
            # trained NMF-model
            'nmf': Pipeline([('nmf', nmf)]),
            # decomposition matrix V, and approximation matrices W and H
            'V': V, 'W': W, 'H': pd.DataFrame(H),
            # timestamps for V
            'V_timestamps': timestamps,
            # reconstructed decomposition matrix
            'V_reconstructed': V_reconstructed,
            # coefficient of determination (average)
            'R2_mean': r2_score(V, V_reconstructed, multioutput='uniform_average'),
            # coefficient of determination (per sample)
            'R2': r2_score(V.T, V_reconstructed.T, multioutput='raw_values'),
            # coefficient of determination (per feature)
            'R2_feature': r2_score(V, V_reconstructed, multioutput='raw_values'),
            # mean squared error (average)
            'MSE_mean': mean_squared_error(V, V_reconstructed),
            # mean squared error (per sample)
            'MSE': mean_squared_error(V.T, V_reconstructed.T, multioutput='raw_values'),
            # reconstruction error expressed as Frobenius norm
            'reconstruction_error': nmf.reconstruction_err_
        }

        list_models.append(model_dict)

        # Update previous_W and previous_H for the next iteration
        previous_W = W
        previous_H = H
    list_models.reverse()
    df_models = pd.DataFrame(list_models)
    return df_models


def calculate_distances_per_measurement_period(measurement_period, fingerprints):

    df_dist_ = []
    for idx, row in measurement_period.iterrows():
        for om in fingerprints:
            weights = row['W']
            fingerprint = fingerprints[om]
            tmp = {
                'idx': idx,
                'data': row,
                'om': om,
                'cosine_distance': distance_metrics.cosine_distance(weights, fingerprint),
                'manhattan_distance': distance_metrics.manhattan_distance(weights, fingerprint),
            }
            df_dist_.append(tmp)
    df_dist_ = pd.DataFrame(df_dist_)
    return df_dist_


def _extract_vibration_weights_per_measurement_period(measurement_periods, col_names, model,
                                                      verbose=False):
    Ws = []
    for period in tqdm(measurement_periods, disable=not verbose,
                       desc='Extracting vibration weights per measurement period'):
        assert len(period) == 3, 'should have exactly 3 directions per measurement period'
        band_column_names = period.columns[period.columns.str.contains('band_')]
        V = period.set_index(['direction'])[band_column_names]  # already normalized
        W = model.nmf.transform(V.to_numpy())
        W = pd.DataFrame(W, columns=col_names)
        Ws.append({
            'unique_sample_id': period.unique_sample_id.unique()[0],
            'V_normalized': V,
            'W': W
        })
    return pd.DataFrame(Ws)


def derive_df_vib(data, f):
    psds = []
    rpms = []
    torques = []
    directions = []
    sample_ids = []
    unique_sample_ids = []
    for x in data:
        rpm = x['rpm']
        torque = x['torque']
        sample_id = x['sample_id']
        for key in ['psd_x', 'psd_y', 'psd_z']:
            psds.append(x[key])
            directions.append(key)
            rpms.append(rpm)
            torques.append(torque)
            sample_ids.append(sample_id)
            unique_sample_ids.append(x['unique_sample_id'])

    df_vib = pd.DataFrame(psds, columns=f)  # !!!
    df_vib['rotational speed [RPM]'] = rpms
    df_vib['torque [Nm]'] = torques
    df_vib['direction'] = directions
    df_vib['sample_id'] = sample_ids
    df_vib['unique_sample_id'] = unique_sample_ids
    return df_vib


def _extract_frequency_bands_for_single_sample(vibrations, velocity_in_rpm, lower_bounds=None, upper_bounds=None,
                                               window_boundaries=(9, 11), order_to_scale_peak_to=10,
                                               correct_orders=True, agg_func=np.mean):
    # set default arguments
    lower_bounds = np.arange(0.25, 30.25, 0.5) if lower_bounds is None else lower_bounds
    upper_bounds = np.arange(0.75, 30.75, 0.5) if upper_bounds is None else upper_bounds
    assert len(lower_bounds) == len(upper_bounds)

    # extract orders with velocity
    velocity_in_Hz = velocity_in_rpm / 60
    orders = vibrations.index.astype(float) / velocity_in_Hz

    # correct orders based on set peak
    if correct_orders:
        selected_frequency_band = vibrations[(orders >= window_boundaries[0]) & (orders <= window_boundaries[1])]
        if len(selected_frequency_band) > 0:
            argmax = np.argmax(selected_frequency_band)
            order_of_uncorrected_peak = orders[(orders >= window_boundaries[0]) &
                                               (orders <= window_boundaries[1])][argmax]
            c = order_to_scale_peak_to / order_of_uncorrected_peak
        else:
            print('Could not find any values in window boundaries. Going to set c to 1.')
            c = 1
        orders = c * orders

    # aggregate over frequency bands
    def aggregate_over_fband(_lb, _ub):
        return agg_func(vibrations[(orders >= _lb) & (orders <= _ub)])
    frequency_bands = {f'band_{lb}-{ub}': aggregate_over_fband(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)}

    # calculate number of bins per aggregated order band
    def n_per_fband(_lb, _ub):
        return len(vibrations[(orders >= _lb) & (orders <= _ub)])
    n_per_frequency_bands = {f'n_{lb}-{ub}': n_per_fband(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)}

    return frequency_bands, n_per_frequency_bands


def _automated_frequency_band_extraction(df, verbose=True, freq_values=None, rpm_name='rotational speed [RPM]',
                                         min_rpm=10, expected_freq_columns=6397, **kwargs):
    """ Given a dataframe with frequency values measured in Hz and a rotational speed, convert those to orders.

    :param df: Pandas dataframe.
    :param verbose: If True, show progress bar. Default: True.
    :param freq_values: Names of columns in df that contain vibration measures. Default: Regex pattern '\d*\.\d*'.
    :param rpm_name: Name of colum with rotational speed. Default: 'rotational speed [RPM]'.
    :param min_rpm: Vibration measurements below the minimum rotational speed [RPM] are not converted. Default: 10.
    :param kwargs: Keyword arguments to be sent to `extract_frequency_bands_for_single_sample()` (see above).
    :return: Pandas dataframe with converted frequency bands.
    """
    freq_values = df.columns[df.columns.str.contains('^\d*\.\d*', na=True)] if freq_values is None else freq_values
    m = f'We expect {expected_freq_columns} frequency values, got {len(freq_values)} instead; did the dataset change?'
    assert len(freq_values) == expected_freq_columns, m
    frequency_bands = []
    for idx, row in tqdm(df.iterrows(), total=len(df), disable=(not verbose)):
        velocity_in_rpm = row[rpm_name]
        if velocity_in_rpm > min_rpm:
            vibrations = row[freq_values]
            fbands, nfbands = _extract_frequency_bands_for_single_sample(vibrations, velocity_in_rpm, **kwargs)
            frequency_bands.append({'index': idx, **fbands, **nfbands})
    frequency_bands = pd.DataFrame(frequency_bands).set_index('index')
    return frequency_bands


def derive_df_orders(df_vib, setup, f, verbose=True):
    lower_bounds = np.arange(setup['start'], setup['stop'], setup['window_steps'])
    upper_bounds = np.arange((setup['start'] + setup['window_size']), (setup['stop'] + setup['window_size']),
                             setup['window_steps'])
    n_windows = setup['n_windows']
    n_lower_bounds = len(lower_bounds)
    n_upper_bounds = len(upper_bounds)
    m = f'Number of windows ({n_windows}), lower bounds ({n_lower_bounds}) and upper bounds ({n_upper_bounds}) ' \
        f'do not match.'
    assert n_windows == n_lower_bounds == n_upper_bounds, m
    df_orders = _automated_frequency_band_extraction(df_vib, verbose=verbose, rpm_name='rotational speed [RPM]',
                                                     min_rpm=10, correct_orders=False, window_boundaries=(19, 23),
                                                     order_to_scale_peak_to=21, lower_bounds=lower_bounds,
                                                     upper_bounds=upper_bounds, agg_func=np.sum,
                                                     expected_freq_columns=f.shape[0])
    meta_data = df_vib[['rotational speed [RPM]', 'torque [Nm]', 'direction', 'sample_id', 'unique_sample_id']]
    return df_orders, meta_data


def get_df_W_offline_and_online(_df_V_train, meta_data_train, meta_data_test, model, df_orders_test,
                                df_operating_modes):
    # extract train vibration measurement periods

    df_V_train = _df_V_train.copy()

    df_V_train[['unique_sample_id', 'direction']] = meta_data_train[['unique_sample_id', 'direction']]
    train_vibration_measurement_periods = []
    for sample_id, group in df_V_train.groupby('unique_sample_id'):
        train_vibration_measurement_periods.append(group)

    # extract test vibration measurement periods

    n_components = model.W.shape[-1]
    W_train = model.W.reshape(-1, n_components)
    df_W_train = pd.DataFrame(W_train)
    df_W_train.index = df_V_train.index
    df_W_train['direction'] = meta_data_train['direction']

    # add operating mode (OM)
    df_W_train_with_OM = pd.merge(df_W_train, meta_data_train.drop(columns=['direction']), left_index=True,
                                  right_index=True)
    # df_W_train_with_OM['cluster_label_unique'] = df_W_train_with_OM.groupby(
    #     ['rotational speed [RPM]', 'torque [Nm]']).ngroup()

    df_W_train_with_OM['cluster_label_unique'] = df_W_train_with_OM[['rotational speed [RPM]', 'torque [Nm]']].\
        apply(lambda x: int(df_operating_modes.loc[int(x[1]), int(x[0])].strip('OM ')), axis=1)

    # print(df_W_train_with_OM['cluster_label_unique'])

    cols_ = df_V_train.columns
    band_cols = cols_[cols_.str.contains('band')].tolist()

    df_V_test_normalized = normalize_along_rows(df_orders_test, band_cols)
    df_ = df_V_test_normalized
    df_[['sample_id', 'unique_sample_id', 'direction']] = meta_data_test[['sample_id', 'unique_sample_id', 'direction']]
    test_vibration_measurement_periods = []
    test_vibration_measurement_periods_meta_data = []
    n_index_errors = 0
    for unique_sample_id, group in df_.groupby('unique_sample_id'):
        rpm = meta_data_test[meta_data_test['unique_sample_id'] ==
                             unique_sample_id]['rotational speed [RPM]'].unique()[0]
        torque = meta_data_test[meta_data_test['unique_sample_id'] == unique_sample_id]['torque [Nm]'].unique()[0]
        try:
            om = df_operating_modes.loc[torque, rpm]
        except IndexError:
            n_index_errors += 1
            om = -1
        measurement_period = {'start': 'unknown',
                              'stop': 'unknown',
                              'group': group,
                              'unique_sample_id': unique_sample_id,
                              'rpm': rpm,
                              'torque': torque,
                              'unique_cluster_label': om}
        test_vibration_measurement_periods.append(group)
        test_vibration_measurement_periods_meta_data.append(measurement_period)

    component_cols = list(range(n_components))
    grouping_vars = ['direction', 'cluster_label_unique']
    df_ = df_W_train_with_OM[component_cols + grouping_vars].copy()
    fingerprints = {
        om: om_group.groupby(['direction']).mean().drop(columns=['cluster_label_unique']) for om, om_group in
        df_.groupby('cluster_label_unique')
    }

    # print(fingerprints.keys()[0])

    df_W_offline = _extract_vibration_weights_per_measurement_period(train_vibration_measurement_periods,
                                                                     fingerprints[list(fingerprints.keys())[0]].columns,
                                                                     model)
    df_W_online = _extract_vibration_weights_per_measurement_period(test_vibration_measurement_periods,
                                                                    fingerprints[list(fingerprints.keys())[0]].columns,
                                                                    model)
    return df_W_offline, df_W_online, fingerprints, test_vibration_measurement_periods_meta_data


def get_pivot_table(df_W_online, fingerprints, test_vibration_measurement_periods_meta_data):
    df_dist_online = calculate_distances_per_measurement_period(df_W_online, fingerprints=fingerprints)

    df_cosine = df_dist_online[['idx', 'om', 'cosine_distance']].pivot(index='idx', columns='om',
                                                                       values='cosine_distance')
    # assign the corresponding operating mode to the given row (if known), else, assign -1
    df_cosine[['rpm', 'torque', 'unique_cluster_label']] = pd.DataFrame(test_vibration_measurement_periods_meta_data)[
        ['rpm', 'torque', 'unique_cluster_label']]

    distance_to_own_cluster_center = []
    for idx, row in df_cosine.iterrows():
        om = int(row['unique_cluster_label'].strip('OM '))
        if om in row.index:
            distance_to_own_cluster_center.append(row[om])
        else:
            distance_to_own_cluster_center.append(np.nan)
    df_cosine['distance_to_own_cluster_center'] = distance_to_own_cluster_center
    df_cosine['pitting'] = df_W_online['unique_sample_id'].str.contains(f'pitting_level_')
    df_cosine['pitting_level'] = df_W_online['unique_sample_id'].str.extract(r'pitting_level_(\d)')
    df_cosine['pitting_level'] = df_cosine['pitting_level'].fillna(0).astype(int)
    return df_cosine
