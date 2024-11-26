import numpy as np
import pandas as pd
import pytest

from modelbase2.fit import steady_state, time_series
from modelbase2.integrators import DefaultIntegrator
from modelbase2.model import Model


@pytest.fixture
def model():
    # Mock model with necessary methods and attributes
    class MockModel(Model):
        def __init__(self):
            self.parameters = {}

        def update_parameters(self, params):
            self.parameters.update(params)
            return self

    return MockModel()


@pytest.fixture
def data_steady_state():
    return pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])


@pytest.fixture
def data_time_series():
    return pd.DataFrame(
        {
            "A": [1.0, 1.5, 2.0],
            "B": [2.0, 2.5, 3.0],
            "C": [3.0, 3.5, 4.0],
        },
        index=[0, 1, 2],
    )


@pytest.fixture
def initial_params():
    return {"param1": 0.1, "param2": 0.2, "param3": 0.3}


@pytest.fixture
def initial_conditions():
    return {"A": 1.0, "B": 2.0, "C": 3.0}


def test_steady_state(model, data_steady_state, initial_params, initial_conditions):
    result = steady_state(
        model=model,
        p0=initial_params,
        data=data_steady_state,
        y0=initial_conditions,
        integrator=DefaultIntegrator,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == set(initial_params.keys())


def test_time_series(model, data_time_series, initial_params, initial_conditions):
    result = time_series(
        model=model,
        p0=initial_params,
        data=data_time_series,
        y0=initial_conditions,
        integrator=DefaultIntegrator,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == set(initial_params.keys())
