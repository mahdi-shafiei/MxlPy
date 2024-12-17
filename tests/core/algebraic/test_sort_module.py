# type: ignore

from __future__ import annotations

from modelbase.ode import Model


def dummy_func(*args):
    pass


def test_sort_algebraic_modules_add():
    m = Model()
    m.add_compounds(["x"])
    m.add_algebraic_module(
        module_name="m1",
        function=dummy_func,
        derived_compounds=["A"],
        compounds=["x"],
    )
    assert m._algebraic_module_order == ["m1"]

    m.add_algebraic_module(
        module_name="m2",
        function=dummy_func,
        compounds=["A"],
        derived_compounds=["B"],
    )
    assert m._algebraic_module_order == ["m1", "m2"]

    m.add_algebraic_module(
        module_name="m3",
        function=dummy_func,
        compounds=["A", "B"],
        derived_compounds=["C"],
    )
    assert m._algebraic_module_order == ["m1", "m2", "m3"]
    m.add_algebraic_module(
        module_name="m4",
        function=dummy_func,
        derived_compounds=["D"],
    )
    assert m._algebraic_module_order == ["m1", "m2", "m3", "m4"]


def test_sort_algebraic_modules_update():
    m = Model()
    m.add_compounds(["x"])
    m.add_algebraic_module(
        module_name="m1",
        function=dummy_func,
        derived_compounds=["A"],
        compounds=["x"],
    )
    m.add_algebraic_module(
        module_name="m2",
        function=dummy_func,
        derived_compounds=["B"],
        compounds=["A"],
    )
    m.add_algebraic_module(
        module_name="m3",
        function=dummy_func,
        derived_compounds=["C"],
        compounds=["x"],
    )
    assert m._algebraic_module_order == ["m1", "m2", "m3"]

    # Now make m1 dependent on m3
    m.update_algebraic_module(module_name="m1", compounds=["C"])
    assert m._algebraic_module_order == ["m3", "m1", "m2"]


def test_sort_algebraic_modules_remove():
    m = Model()
    m.add_compounds(["x"])
    m.add_algebraic_module(
        module_name="m1",
        function=dummy_func,
        derived_compounds=["A"],
        compounds=["x"],
    )
    m.add_algebraic_module(
        module_name="m2",
        function=dummy_func,
        derived_compounds=["B"],
        compounds=["A"],
    )
    m.add_algebraic_module(
        module_name="m3",
        function=dummy_func,
        derived_compounds=["C"],
        compounds=["x"],
    )
    assert m._algebraic_module_order == ["m1", "m2", "m3"]

    # Now remove m2
    m.remove_algebraic_module(module_name="m2")
    assert m._algebraic_module_order == ["m1", "m3"]


def test_sort_from_args_correct_order():
    m = Model()
    m.add_compounds(["A"])
    m.add_algebraic_module_from_args(
        module_name="b",
        function=lambda x: x,
        args=["A"],
        derived_compounds=["B"],
    )
    m.add_algebraic_module_from_args(
        module_name="c",
        function=lambda x: x,
        args=["B"],
        derived_compounds=["C"],
    )
    assert m._algebraic_module_order == ["b", "c"]


def test_sort_from_args_wrong_order():
    m = Model()
    m.add_compounds(["A"])
    m.add_algebraic_module_from_args(
        module_name="c",
        function=lambda x: x,
        args=["B"],
        derived_compounds=["C"],
    )
    m.add_algebraic_module_from_args(
        module_name="b",
        function=lambda x: x,
        args=["A"],
        derived_compounds=["B"],
    )
    assert m._algebraic_module_order == ["b", "c"]
