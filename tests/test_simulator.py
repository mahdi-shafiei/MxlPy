import numpy as np
import pandas as pd
import pytest

from modelbase2.simulator import _normalise_split_results


def test_normalise_split_results_with_scalar() -> None:
    results = [
        pd.DataFrame(np.array([[1, 2], [3, 4]])),
        pd.DataFrame(np.array([[5, 6], [7, 8]])),
    ]
    normalise = 2
    expected = [
        pd.DataFrame(np.array([[0.5, 1], [1.5, 2]])),
        pd.DataFrame(np.array([[2.5, 3], [3.5, 4]])),
    ]
    output = _normalise_split_results(results, normalise)
    for out, exp in zip(output, expected):
        pd.testing.assert_frame_equal(out, exp)


def test_normalise_split_results_with_array() -> None:
    results = [
        pd.DataFrame(np.array([[1, 2], [3, 4]])),
        pd.DataFrame(np.array([[5, 6], [7, 8]])),
    ]
    normalise = [2, 4]
    expected = [
        pd.DataFrame(np.array([[0.5, 1], [0.75, 1]])),
        pd.DataFrame(np.array([[1.25, 1.5], [1.75, 2]])),
    ]
    output = _normalise_split_results(results, normalise)
    for out, exp in zip(output, expected):
        pd.testing.assert_frame_equal(out, exp)


def test_normalise_split_results_with_mismatched_lengths() -> None:
    results = [
        pd.DataFrame(np.array([[1, 2], [3, 4]])),
        pd.DataFrame(np.array([[5, 6], [7, 8]])),
    ]
    normalise = [2, 4, 6, 8]
    expected = [
        pd.DataFrame(np.array([[0.5, 1], [0.75, 1]])),
        pd.DataFrame(np.array([[0.625, 0.75], [0.875, 1]])),
    ]
    output = _normalise_split_results(results, normalise)
    for out, exp in zip(output, expected):
        pd.testing.assert_frame_equal(out, exp)


def test_normalise_split_results_with_empty_results() -> None:
    results = []
    normalise = 2
    expected = []
    output = _normalise_split_results(results, normalise)
    assert output == expected


def test_normalise_split_results_with_empty_normalise() -> None:
    results = [
        pd.DataFrame(np.array([[1, 2], [3, 4]])),
        pd.DataFrame(np.array([[5, 6], [7, 8]])),
    ]
    normalise = []
    with pytest.raises(ValueError):
        _normalise_split_results(results, normalise)
