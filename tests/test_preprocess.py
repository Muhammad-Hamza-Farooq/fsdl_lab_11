import numpy as np
import pytest

from preprocess import normalize


def test_normalize_mean():
    data = np.array([1, 2, 3, 4, 5])
    normalized = normalize(data)

    assert round(normalized.mean(), 5) == 0


def test_normalize_nan():
    data = np.array([1.0, 2.0, np.nan, 4.0])
    result = normalize(data)
    assert np.isnan(result).any()


def test_normalize_empty():
    data = np.array([])
    with pytest.raises(ValueError):
        normalize(data)


def test_normalize_single_value():
    data = np.array([42.0])
    with pytest.raises(ValueError, match="single-value"):
        normalize(data)
