import numpy as np


def _validate_data(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Invalid Data Shape"


def r2_score(y_true, y_pred):
    _validate_data(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def mse_score(y_true, y_pred):
    _validate_data(y_true, y_pred)
    n = len(y_true)
    mse = 1 / n * np.sum(y_true - y_pred) ** 2
    return mse


def rmse_score(y_true, y_pred):
    return np.sqrt(mse_score(y_true, y_pred))


def mae_score(y_true, y_pred):
    _validate_data(y_true, y_pred)
    n = len(y_true)
    mae = 1 / n * np.sum(abs(y_true - y_pred))
    return mae


def mape(y_true, y_pred):
    _validate_data(y_true, y_pred)
    n = len(y_true)
    mape = 1 / n * np.sum(abs((y_true - y_pred) / y_true))
    return mape
