from abc import abstractmethod
from typing import Literal

import numpy as np

from models.base import BaseModelLinear


class LinearRegression(BaseModelLinear):
    def __init__(
        self,
        type: Literal["analyt", "grad"] = "analyt",
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        bias: bool = True,
    ):
        super().__init__(n_iterations, learning_rate)
        self.type = type
        self.bias = bias

    def _anal(self, X, y):
        return np.linalg.inv((X.T @ X)) @ X.T @ y

    def _add_bias(self, X: np.ndarray):
        if self.bias:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._validate_data(X, y)
        X_b = self._add_bias(X)
        n, m = X_b.shape

        if self.type == "analyt":
            self.weights = self._anal(X_b, y)
        elif self.type == "grad":
            self._grad(X_b, y, n, m)

    def _grad_loss(self, X, y, y_pred, n):
        return 2 / n * X.T @ (y_pred - y)

    def _grad(self, X: np.ndarray, y: np.ndarray, n, m):
        self.weights = np.zeros(m)
        for _ in range(self.n_iterations):
            y_pred = X @ self.weights
            grad = self._grad_loss(X, y, y_pred, n)
            self.weights = self.weights - self.learning_rate * grad

    def predict(self, X):
        X = self._add_bias(X)
        y_pred = X @ self.weights
        return y_pred


class RidgeRegression(LinearRegression):
    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _anal(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        I_matrix = np.eye(X.shape[1])
        I_matrix[0, 0] = 0
        return np.linalg.inv((X.T @ X + self.alpha * I_matrix)) @ X.T @ y

    def _grad_loss(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, n
    ) -> np.ndarray:
        grad = 2 / n * (X.T @ (y_pred - y))
        l2 = 2 * self.alpha / n * self.weights
        l2[0] = 0
        return grad + l2


class LassoRegression(LinearRegression):
    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _grad_loss(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, n
    ) -> np.ndarray:
        grad = 2 / n * X.T @ (y_pred - y)
        l1 = self.alpha * np.sign(self.weights)
        l1[0] = 0
        return grad + l1


class ElasticNet(LinearRegression):
    def __init__(self, alpha: float = 1.0, l1_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def _grad_loss(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, n
    ) -> np.ndarray:
        grad = (1 / n) * X.T @ (y_pred - y)
        l2 = ((1 - self.l1_ratio) * self.alpha / n) * self.weights
        l2[0] = 0
        l1 = self.l1_ratio * self.alpha * np.sign(self.weights)
        l1[0] = 0

        return grad + l1 + l2


class LogisticRegressionBase(BaseModelLinear):
    @abstractmethod
    def predict_proba(self, X): ...


class BinaryLogisticRegression(LogisticRegressionBase):
    def __init__(self, threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _add_bias(self, X: np.ndarray):
        if self.bias:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._validate_data(X, y)
        X_b = self._add_bias(X)
        n, m = X_b.shape
        self._grad(X_b, y, n, m)

    def _grad_loss_bin(self, X, y, y_pred, n):
        errors = y_pred - y
        return np.dot(X.T, errors) / n

    def _grad(self, X: np.ndarray, y: np.ndarray, n, m):
        self.weights = np.zeros(m)
        for _ in range(self.n_iterations):
            y_pred = self._sigmoid(X @ self.weights)
            grad = self._grad_loss_bin(X, y, y_pred, n)

            self.weights = self.weights - self.learning_rate * grad

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        y_pred = self.predict_proba(X)
        y_pred = np.where(y_pred > self.threshold, 1, 0)
        return y_pred

    def predict_proba(self, X):
        X = self._add_bias(X)
        X = X @ self.weights
        y_pred = self._sigmoid(X)
        return y_pred


class MultiClassLogisticRegression(LogisticRegressionBase):
    def __init__(self, threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _add_bias(self, X: np.ndarray):
        if self.bias:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def _onehot(self, y):
        # TODO: хз предусмотреть случай когда класс не от 0
        return np.eye(self.num_classes)[y]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._validate_data(X, y)
        self.num_classes = len(np.unique(y))
        X_b = self._add_bias(X)
        n, m = X_b.shape
        self._grad(X_b, y, n, m)

    def _grad_loss_mult(self, X, y, y_pred, n):
        y = self._onehot(y)
        errors = y_pred - y
        return np.dot(X.T, errors) / n

    def _grad(self, X: np.ndarray, y: np.ndarray, n, m):
        self.weights = np.zeros((m, self.num_classes))
        for _ in range(self.n_iterations):
            y_pred = self._softmax(X @ self.weights)
            grad = self._grad_loss_mult(X, y, y_pred, n)

            self.weights = self.weights - self.learning_rate * grad

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def predict(self, X):
        y_pred = self.predict_proba(X)
        y_pred = y_pred.argmax(axis=1)
        return y_pred

    def predict_proba(self, X):
        X = self._add_bias(X)
        X = X @ self.weights
        y_pred = self._softmax(X)
        return y_pred


class LogisticRegression(LogisticRegressionBase):
    def __new__(
        self, type: Literal["binary", "multiclass"] = "multiclass", *args, **kwargs
    ):
        if type == "binary":
            return BinaryLogisticRegression(*args, **kwargs)
        elif type == "multiclass":
            return MultiClassLogisticRegression(*args, **kwargs)
        else:
            raise ValueError(
                "Invalid LogisticRegression type, available types: ['binary', 'multiclass']"
            )
