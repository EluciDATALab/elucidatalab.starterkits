# ©, 2022, Sirris
# owner: FFNG

import os
import glob
from starterkits.starterkit_1_2_2.data_handler import BASE_PATH_HEALTHY
import pandas as pd
import numpy as np
from sklearn.metrics import auc


def calculate_roc_characteristics(df_):
    df_ = df_.sort_values(by='distance_to_own_cluster_center', ascending=True)

    # Initialize variables to store ROC curve values
    fpr = []
    tpr = []

    for threshold in df_['distance_to_own_cluster_center']:
        df_['predicted_anomaly'] = df_['distance_to_own_cluster_center'] >= threshold

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        true_positives = df_[(df_['pitting'] == 1) & (df_['predicted_anomaly'] == 1)].shape[0]
        false_positives = df_[(df_['pitting'] == 0) & (df_['predicted_anomaly'] == 1)].shape[0]
        true_negatives = df_[(df_['pitting'] == 0) & (df_['predicted_anomaly'] == 0)].shape[0]
        false_negatives = df_[(df_['pitting'] == 1) & (df_['predicted_anomaly'] == 0)].shape[0]

        tpr.append(true_positives / (true_positives + false_negatives))
        fpr.append(false_positives / (false_positives + true_negatives))

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def calc_tpr_at_fpr_threshold(tpr, fpr, threshold=0.1):
    # sort tpr and fpr such that they are in ascending order
    if (fpr[0] > fpr[-1]) or (tpr[0] > tpr[-1]):
        assert tpr[0] > tpr[-1] and fpr[0] > fpr[-1]
        tpr = list(reversed(tpr))
        fpr = list(reversed(fpr))
    try:
        idx = next(i for i, value in enumerate(fpr) if value > threshold)
    except StopIteration:
        idx = 0
    tpr_at_fpr = tpr[idx]
    return tpr_at_fpr


def calc_fpr_at_tpr_threshold(tpr, fpr, threshold=0.1):
    return calc_tpr_at_fpr_threshold(tpr=fpr, fpr=tpr, threshold=threshold)


def get_operating_modes():
    file_names_healthy = glob.glob(os.path.join(BASE_PATH_HEALTHY, '*.txt'))
    rpms = [int(_f.split('/')[-1].split('_')[0].strip('V')) for _f in file_names_healthy]
    torques = [int(_f.split('/')[-1].split('_')[1].strip('N')) for _f in file_names_healthy]
    df_operating_modes = pd.DataFrame(columns=sorted(np.unique(rpms)), index=sorted(np.unique(torques)), data='')
    counter = 1
    for torque in sorted(np.unique(torques)):
        for rpm in sorted(np.unique(rpms)):
            runs = glob.glob(os.path.join(BASE_PATH_HEALTHY, f'V{rpm}_{torque}N_*.txt'))
            runs = [int(run.split('/')[-1].split('_')[2].strip('.txt')) for run in runs]
            if len(runs) > 0:
                df_operating_modes.loc[torque, rpm] = f'OM {counter}'
                counter += 1
    return df_operating_modes
