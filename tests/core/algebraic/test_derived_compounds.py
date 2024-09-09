# type: ignore

from __future__ import annotations

from modelbase2.ode import Model


def test_derived_compounds():
    model = Model()
    model.add_compound("A")
    model.add_algebraic_module(
        module_name="mod1",
        function=lambda s, keq: (s / (1 + keq), (s * keq / (1 + keq))),
        compounds=["A"],
        derived_compounds=["x", "y"],
        parameters=["keq"],
    )
    assert model.derived_compounds == model.get_derived_compounds()


def test_get_all_compounds():
    model = Model()
    model.add_compound("A")
    model.add_algebraic_module(
        module_name="mod1",
        function=lambda s, keq: (s / (1 + keq), (s * keq / (1 + keq))),
        compounds=["A"],
        derived_compounds=["x", "y"],
        parameters=["keq"],
    )
    assert model.get_all_compounds() == ["A", "x", "y"]
