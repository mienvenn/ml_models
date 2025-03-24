from typing import Iterable, Literal

import numpy as np

from models.base import BaseModelTree


class Node:
    def __init__(self, X, feature, threshold):
        self.left = X[:, feature] <= threshold
        self.right = ~self.left


# TODO: выбирать не по уникальным значениям, а по среднему среди уникальынх


class DecisionTree(BaseModelTree):
    def __init__(self, min_samples_split: int = 1, max_depth: int = 4):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[float]) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        self._validate_data(X, y)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        if len(y) < self.min_samples_split or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return self._result(y)

        best_split = self._find_best_split(X, y)

        if best_split is None:
            return self._result(y)

        val = Node(X, best_split["feature"], best_split["threshold"])
        left = val.left
        right = val.right

        left_subtree = self._build_tree(X[left], y[left], depth + 1)
        right_subtree = self._build_tree(X[right], y[right], depth + 1)

        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_subtree,
            "right": right_subtree,
        }

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
        best_mse = float("inf")
        best_split = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds[:-1]:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                mse = self._ig(y[left_mask], y[right_mask])

                if mse < best_mse:
                    best_mse = mse
                    best_split = {"feature": feature, "threshold": threshold}

        return best_split

    def _predict_sample(self, sample: np.ndarray, tree: dict | float):
        if isinstance(tree, dict):
            if sample[tree["feature"]] <= tree["threshold"]:
                return self._predict_sample(sample, tree["left"])
            else:
                return self._predict_sample(sample, tree["right"])
        return tree

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray[float]:
        X = np.asarray(X)
        return np.array([self._predict_sample(sample, self.tree) for sample in X])


class DecisionTreeRegressor(DecisionTree):
    def _result(self, y: np.ndarray) -> float:
        return np.mean(y)

    def _ig(self, left: np.ndarray, right: np.ndarray) -> float:
        ig_left = np.sum((left - np.mean(left)) ** 2)
        ig_right = np.sum((right - np.mean(right)) ** 2)
        n = len(right) + len(left)
        ig = (ig_left + ig_right) / n
        return ig


class DecisionTreeClassifier(DecisionTree):
    def __init__(
        self, criterion: Literal["gini", "entropy"] = "entropy", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        str_to_criterion = {"gini": self._gini, "entropy": self._entropy}
        self.criterion = str_to_criterion[criterion]

    def _result(self, y: np.ndarray) -> float:
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _gini(self, p_left: np.ndarray, p_right: np.ndarray) -> tuple[float, float]:
        ig_left = 1 - np.sum(p_left**2)
        ig_right = 1 - np.sum(p_right**2)
        return ig_left, ig_right

    def _entropy(self, p_left: np.ndarray, p_right: np.ndarray) -> tuple[float, float]:
        ig_left = -np.sum(p_left * np.log(p_left))
        ig_right = -np.sum(p_right * np.log(p_right))
        return ig_left, ig_right

    def _ig(self, left: np.ndarray, right: np.ndarray) -> float:
        p_left = np.unique(left, return_counts=True)[1] / len(left)
        p_right = np.unique(right, return_counts=True)[1] / len(right)
        ig_left, ig_right = self.criterion(p_left, p_right)
        n = len(right) + len(left)
        ig = (len(left) * ig_left + len(right) * ig_right) / n
        return ig
