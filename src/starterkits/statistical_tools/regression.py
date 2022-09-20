# ©, 2019, Sirris
# owner: HCAB

"""Functions to assist with regression evaluation"""

import numpy as np


def rmse(obs, pred):
    """
    Calculate RMSE

    :param obs: list/array. observed
    :param pred: list/array. predicted
    :return: rmse
    """
    return np.sqrt(np.sum(np.power(obs - pred, 2)) / len(obs))


def nrmse(obs, pred):
    """
    Calculate NMRSE

    :param obs: list/array. observed
    :param pred: list/array. predicted
    :return: nrmse
    """
    return np.sqrt(np.sum(np.power(obs - pred, 2)) / len(obs)) / obs.mean()


def rae(obs, pred):
    """
    Calculate RAE

    :param obs: list/array. observed
    :param pred: list/array. predicted
    :return: rae
    """
    return np.sum(np.abs(obs - pred)) / np.sum(np.abs(obs.mean() - pred))


def rrse(obs, pred):
    """
    Calculate RRSE

    :param obs: list/array. observed
    :param pred: list/array. predicted
    :return: rrse
    """
    return np.sqrt(np.sum(np.power(obs - pred, 2)) / np.sum(np.power(obs.mean() - pred, 2)))


def mape(obs, pred):
    """
    Calculate MAPE

    :param obs: list/array. observed
    :param pred: list/array. predicted
    :return: mape
    """
    return np.sum(np.abs((obs - pred) / obs)) / len(obs)
