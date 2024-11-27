from typing import Self

import pandas as pd
import pytest

from modelbase2 import fit
from modelbase2.integrators import DefaultIntegrator
from modelbase2.model import Model


class MockModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def update_parameters(self, parameters: dict[str, float]) -> Self:
        self._parameters.update(parameters)
        return self


@pytest.fixture
def model() -> MockModel:
    return MockModel()


def test_steady_state_basic(model: Model) -> None:
    p0 = {"param1": 1.0, "param2": 2.0}
    data = pd.Series([1.0, 2.0], index=["species1", "species2"])
    y0 = {"species1": 0.5, "species2": 1.5}

    result = fit.steady_state(model, p0, data, y0)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(p0.keys())


def test_steady_state_no_initial_conditions(model: Model) -> None:
    p0 = {"param1": 1.0, "param2": 2.0}
    data = pd.Series([1.0, 2.0], index=["species1", "species2"])

    result = fit.steady_state(model, p0, data)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(p0.keys())


def test_steady_state_custom_integrator(model: Model) -> None:
    p0 = {"param1": 1.0, "param2": 2.0}
    data = pd.Series([1.0, 2.0], index=["species1", "species2"])
    y0 = {"species1": 0.5, "species2": 1.5}

    class CustomIntegrator(DefaultIntegrator):
        pass

    result = fit.steady_state(model, p0, data, y0, integrator=CustomIntegrator)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(p0.keys())
