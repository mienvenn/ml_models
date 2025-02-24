from typing import Iterable, Literal

import numpy as np


def _validate_data(X, y):
    assert y.shape[0] == X.shape[0], "Invalid Data Shape"


def data_split(X, y, size: Iterable[float]):
    _validate_data(X, y)
    X = np.array(X)
    y = np.array(y)
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    size = np.array([0] + list(size))
    n = len(X)
    size = np.round(n * size)
    size = np.cumsum(size).astype("int")
    shuffled_ind = [ind[size[i] : size[i + 1]] for i in range(len(size) - 1)]
    res = []
    for i in range(len(shuffled_ind)):
        res += [X[shuffled_ind[i]], y[shuffled_ind[i]]]

    return res


class Normalizer:
    def __init__(self, type: Literal["min_max", "standart"] = "min_max"):
        self.type = type
        self.is_fitted_min_max = False
        self.is_fitted_standart = False

    def fit(self, X, columns=None):
        if self.type == "min_max":
            self._fit_min_max(X[:, columns])
            self.is_fitted_min_max = True
        elif self.type == "standart":
            self._fit_standart(X[:, columns])
            self.is_fitted_standart = True

    def transform(self, X, columns):
        X = X.copy()
        if self.type == "min_max":
            assert self.is_fitted_min_max, "Data not fitted"
            res = self._transform_min_max(X[:, columns])
            X[:, columns] = res
        elif self.type == "standart":
            assert self.is_fitted_standart, "Data not fitted"
            res = self._transform_standart(X[:, columns])
            X[:, columns] = res
        return X

    def _fit_min_max(self, X):
        self.min_ = np.min(X)
        self.max_ = np.max(X)

    def _fit_standart(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)

    def _transform_standart(self, X):
        return (X - self.mean) / self.std

    def _transform_min_max(self, X):
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X, columns=None):
        self.fit(X, columns)
        return self.transform(X, columns)
