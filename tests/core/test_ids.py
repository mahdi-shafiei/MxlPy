# type: ignore

from __future__ import annotations

import pytest

from modelbase2.ode import Model


def dummy_function():
    pass


def test_compound_ids():
    m = Model()
    m.add_compound(compound="A")
    assert m._ids == {"A"}
    m.remove_compound(compound="A")
    assert m._ids == set()


def test_parameter_ids():
    m = Model()
    m.add_parameter(parameter_name="A", parameter_value=1)
    assert m._ids == {"A"}
    m.remove_parameter(parameter_name="A")
    assert m._ids == set()


def test_algebraic_module_ids():
    m = Model()
    m.add_algebraic_module(
        module_name="mod1",
        function=dummy_function,
        derived_compounds=["A"],
    )
    assert m._ids == {"A"}
    m.update_algebraic_module(
        module_name="mod1",
        derived_compounds=["B"],
    )
    assert m._ids == {"B"}
    m.remove_algebraic_module(module_name="mod1")
    assert m._ids == set()


def test_id_warnings_compound_parameter():
    m = Model()
    m.add_compound("name")
    assert m._ids == {"name"}

    # First warning because name id is already given
    with pytest.warns(UserWarning):
        m.add_parameter("name", 1)
    assert m._ids == {"name"}

    m.remove_compound("name")
    assert m._ids == set()

    # Second warning, because id was removed with the compound
    with pytest.warns(UserWarning):
        m.remove_parameter("name")


def test_id_warnings_parameter_compound():
    m = Model()
    m.add_parameter("name", 1)
    assert m._ids == {"name"}

    # First warning because name id is already given
    with pytest.warns(UserWarning):
        m.add_compound("name")
    assert m._ids == {"name"}
    m.remove_parameter("name")
    assert m._ids == set()

    # Second warning, because id was removed with the compound
    with pytest.warns(UserWarning):
        m.remove_compound("name")


def test_id_warnings_module_and_others():
    m = Model()
    m.add_algebraic_module(
        module_name="mod1",
        function=dummy_function,
        derived_compounds=["name"],
    )
    with pytest.warns(UserWarning):
        m.add_compound("name")
    with pytest.warns(UserWarning):
        m.add_parameter("name", 1)
