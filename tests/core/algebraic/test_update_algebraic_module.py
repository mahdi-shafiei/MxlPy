# type: ignore

from __future__ import annotations

import pytest
from modelbase.ode import Model


def function1():
    pass


def function2():
    pass


DEFAULT_COMPOUNDS = ["x11", "x12", "x13", "x21", "x22", "x23", "m11", "m12", "m13"]
DEFAULT_PARAMETERS = {"p11": 1, "p12": 2, "p13": 3, "p21": 4, "p22": 5, "p23": 6}
DEFAULT_MODULE = dict(
    function=function1,
    compounds=["x11", "x12"],
    modifiers=["m11", "m12"],
    derived_compounds=["y11", "y12"],
    parameters=["p11", "p12"],
)
DEFAULT_MODEL = Model(
    compounds=DEFAULT_COMPOUNDS,
    parameters=DEFAULT_PARAMETERS,
    algebraic_modules={"mod1": DEFAULT_MODULE},
)


def test_default_model():
    model = DEFAULT_MODEL.copy()
    assert model.get_compounds() == [
        "x11",
        "x12",
        "x13",
        "x21",
        "x22",
        "x23",
        "m11",
        "m12",
        "m13",
    ]
    assert model.get_parameters() == {
        "p11": 1,
        "p12": 2,
        "p13": 3,
        "p21": 4,
        "p22": 5,
        "p23": 6,
    }
    assert model.get_derived_compounds() == ["y11", "y12"]


def test_update_algebraic_module_nothing():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1")
    assert model.get_derived_compounds() == ["y11", "y12"]

    mod = model.algebraic_modules["mod1"]
    assert mod["function"].__name__ == "function1"
    assert mod["compounds"] == ["x11", "x12"]
    assert mod["modifiers"] == ["m11", "m12"]
    assert mod["derived_compounds"] == ["y11", "y12"]
    assert mod["parameters"] == ["p11", "p12"]
    assert mod["args"] == ["x11", "x12", "m11", "m12", "p11", "p12"]


###############################################################################
# Function
###############################################################################


def test_update_algebraic_module_function():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", function=function2)
    mod = model.algebraic_modules["mod1"]
    assert mod["function"].__name__ == "function2"


###############################################################################
# Compounds
###############################################################################


def test_update_algebraic_module_compounds():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", compounds=["x21", "x22"])
    mod = model.algebraic_modules["mod1"]
    assert mod["compounds"] == ["x21", "x22"]
    assert mod["args"] == ["x21", "x22", "m11", "m12", "p11", "p12"]


def test_update_algebraic_module_compounds_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", compounds=["x21"])
    mod = model.algebraic_modules["mod1"]
    assert mod["compounds"] == ["x21"]
    assert mod["args"] == ["x21", "m11", "m12", "p11", "p12"]


def test_update_algebraic_module_compounds_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", compounds=["x21", "x22", "x23"])
    mod = model.algebraic_modules["mod1"]
    assert mod["compounds"] == ["x21", "x22", "x23"]
    assert mod["args"] == ["x21", "x22", "x23", "m11", "m12", "p11", "p12"]


###############################################################################
# Derived compounds
###############################################################################


def test_update_algebraic_module_derived_compounds():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", derived_compounds=["y21", "y22"])
    assert model.get_derived_compounds() == ["y21", "y22"]

    mod = model.algebraic_modules["mod1"]
    assert mod["derived_compounds"] == ["y21", "y22"]


def test_update_algebraic_module_derived_compounds_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", derived_compounds=["y21"])
    assert model.get_derived_compounds() == ["y21"]

    mod = model.algebraic_modules["mod1"]
    assert mod["derived_compounds"] == ["y21"]


def test_update_algebraic_module_derived_compounds_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(
        module_name="mod1", derived_compounds=["y21", "y22", "y23"]
    )
    assert model.get_derived_compounds() == ["y21", "y22", "y23"]

    mod = model.algebraic_modules["mod1"]
    assert mod["derived_compounds"] == ["y21", "y22", "y23"]


###############################################################################
# Modifiers
###############################################################################


def test_update_algebraic_module_modifiers():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", modifiers=["m21", "m22"])

    mod = model.algebraic_modules["mod1"]
    assert mod["modifiers"] == ["m21", "m22"]
    assert mod["args"] == ["x11", "x12", "m21", "m22", "p11", "p12"]


def test_update_algebraic_module_modifiers_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", modifiers=["m21"])

    mod = model.algebraic_modules["mod1"]
    assert mod["modifiers"] == ["m21"]
    assert mod["args"] == ["x11", "x12", "m21", "p11", "p12"]


def test_update_algebraic_module_modifiers_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", modifiers=["m21", "m22", "m23"])

    mod = model.algebraic_modules["mod1"]
    assert mod["modifiers"] == ["m21", "m22", "m23"]
    assert mod["args"] == ["x11", "x12", "m21", "m22", "m23", "p11", "p12"]


###############################################################################
# Parameters
###############################################################################


def test_update_algebraic_module_parameters():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", parameters=["p21", "p22"])

    mod = model.algebraic_modules["mod1"]
    assert mod["parameters"] == ["p21", "p22"]
    assert mod["args"] == ["x11", "x12", "m11", "m12", "p21", "p22"]


def test_update_algebraic_module_parameters_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", parameters=["p21"])

    mod = model.algebraic_modules["mod1"]
    assert mod["parameters"] == ["p21"]
    assert mod["args"] == ["x11", "x12", "m11", "m12", "p21"]


def test_update_algebraic_module_parameters_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(module_name="mod1", parameters=["p21", "p22", "p23"])

    mod = model.algebraic_modules["mod1"]
    assert mod["parameters"] == ["p21", "p22", "p23"]
    assert mod["args"] == ["x11", "x12", "m11", "m12", "p21", "p22", "p23"]


###############################################################################
# Args
###############################################################################


def test_keep_args_on_other_change():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(
        module_name="mod1", args=["p11", "p12", "m12", "m22", "x11", "x22"]
    )
    model.update_algebraic_module(module_name="mod1", function=function2)

    mod = model.algebraic_modules["mod1"]
    assert mod["function"].__name__ == "function2"
    assert mod["args"] == ["p11", "p12", "m12", "m22", "x11", "x22"]


def test_update_algebraic_module_args():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(
        module_name="mod1", args=["p11", "p12", "m12", "m22", "x11", "x22"]
    )

    mod = model.algebraic_modules["mod1"]
    assert mod["args"] == ["p11", "p12", "m12", "m22", "x11", "x22"]


def test_update_algebraic_module_args_on_compound_change():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(
        module_name="mod1",
        compounds=["x21"],
        args=["p11", "p12", "m12", "m22", "x21"],
    )

    mod = model.algebraic_modules["mod1"]
    assert mod["args"] == ["p11", "p12", "m12", "m22", "x21"]


def test_update_algebraic_module_args_on_modifier_change():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(
        module_name="mod1",
        modifiers=["m21"],
        args=["p11", "p12", "m21", "x11", "x22"],
    )

    mod = model.algebraic_modules["mod1"]
    assert mod["args"] == ["p11", "p12", "m21", "x11", "x22"]


def test_update_algebraic_module_args_on_parameter_change():
    model = DEFAULT_MODEL.copy()
    model.update_algebraic_module(
        module_name="mod1",
        parameters=["p21"],
        args=["p21", "m12", "m22", "x11", "x22"],
    )

    mod = model.algebraic_modules["mod1"]
    assert mod["args"] == ["p21", "m12", "m22", "x11", "x22"]
