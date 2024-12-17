# type: ignore

from __future__ import annotations

import pytest
from modelbase.ode import Model


def test_generate_source_code_one_module(multiline_comparison):
    parameters = {"keq": 1}
    model = Model(parameters=parameters)
    model.add_compound("A")
    model.add_algebraic_module(
        module_name="mod1",
        function=lambda s, keq: (s / (1 + keq), (s * keq / (1 + keq))),
        compounds=["A"],
        derived_compounds=["x", "y"],
        parameters=["keq"],
    )

    module_funcs, modules = model._generate_algebraic_modules_source_code()
    expected = ["def mod1(s, keq):", "    return (s / (1 + keq), (s * keq / (1 + keq)))"]
    multiline_comparison(expected, module_funcs)

    expected = [
        "m.add_algebraic_module(",
        "    module_name='mod1',",
        "    function=mod1,",
        "    compounds=['A'],",
        "    derived_compounds=['x', 'y'],",
        "    modifiers=[],",
        "    parameters=['keq'],",
        "    args=['A', 'keq'],",
        ")",
    ]
    multiline_comparison(expected, modules)


def test_generate_source_code_lambda(multiline_comparison):
    parameters = {"keq": 1}
    model = Model(parameters=parameters)
    model.add_compound("A")
    model.add_algebraic_module(
        module_name="mod1",
        function=lambda s, k_eq: (s / (1 + k_eq), s * k_eq / (1 + k_eq)),
        compounds=["A"],
        derived_compounds=["x", "y"],
        parameters=["keq"],
    )

    module_funcs, modules = model._generate_algebraic_modules_source_code()
    expected = [
        "def mod1(s, k_eq):",
        "    return (s / (1 + k_eq), s * k_eq / (1 + k_eq))",
    ]
    multiline_comparison(expected, module_funcs)

    expected = [
        "m.add_algebraic_module(",
        "    module_name='mod1',",
        "    function=mod1,",
        "    compounds=['A'],",
        "    derived_compounds=['x', 'y'],",
        "    modifiers=[],",
        "    parameters=['keq'],",
        "    args=['A', 'keq'],",
        ")",
    ]
    multiline_comparison(expected, modules)
