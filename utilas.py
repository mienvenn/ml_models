import numpy as np


def compute_class_weight(y):
    classes, n_c = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    w = {classes[i]: n / (k * n_c[i]) for i in range(k)}
    return w
