from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class BaseModel(ABC):
    def __init__(self, n_iterations, learning_rate):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...


class LinearRegression(BaseModel):
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

    def _validate_data(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "Invalid Data Shape"

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
        I = np.eye(X.shape[1])
        I[0, 0] = 0
        return np.linalg.inv((X.T @ X + self.alpha * I)) @ X.T @ y

    def _grad_loss(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, n
    ) -> np.ndarray:
        grad = 2 / n * (X.T @ (y_pred - y)) + 2 * self.alpha * self.weights
        grad[0] -= 2 * self.alpha * self.weights[0]
        return grad


class LassoRegression(LinearRegression):
    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _grad_loss(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, n
    ) -> np.ndarray:
        grad = 2 / n * X.T @ (y_pred - y)
        reg = self.alpha * np.sign(self.weights)
        reg[0] = 0
        return grad + reg
