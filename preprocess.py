import numpy as np


def normalize(data):
    if data.size == 0:
        raise ValueError("Cannot normalize an empty array")

    if data.size == 1:
        raise ValueError("Cannot normalize a single-value array (std is zero)")

    return (data - np.mean(data)) / np.std(data)
