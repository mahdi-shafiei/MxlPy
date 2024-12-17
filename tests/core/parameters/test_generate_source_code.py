from __future__ import annotations

from typing import Any, Dict

import pytest
from modelbase.ode import Model


def constant(x: float) -> float:
    return x


def test_generate_constant_parameters_source_code() -> None:
    model = Model()
    kwargs: Dict[str, Any] = {"unit": "mM"}
    model.add_parameter("k_in", 1, **kwargs)
    expected = (
        "m.add_parameters(parameters={'k_in': 1}, meta_info={'k_in': {'unit': 'mM'}})"
    )
    assert expected == model._generate_constant_parameters_source_code(
        include_meta_info=True
    )

    expected = "m.add_parameters(parameters={'k_in': 1})"
    assert expected == model._generate_constant_parameters_source_code(
        include_meta_info=False
    )


def test_derived_generate_constant_parameters_source_code() -> None:
    model = Model()
    model.add_parameter("k", 1)
    model.add_derived_parameter("k_dev", constant, ["k"])

    expected_fn = "def constant(x: float) -> float:\n    return x"
    expected_pars = "m.add_derived_parameter(\n    parameter_name='k_dev',\n    function=constant,\n    parameters=['k'],\n)"
    fns, pars = model._generate_derived_parameters_source_code()
    assert expected_fn == fns
    assert expected_pars == pars
