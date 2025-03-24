from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def _validate_data(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "Invalid Data Shape"

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...


class BaseModelLinear(BaseModel):
    def __init__(
        self, n_iterations: int = 1000, learning_rate: float = 0.01, bias: bool = True
    ):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = bias


class BaseModelTree(BaseModel): ...
