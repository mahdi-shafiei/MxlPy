# type: ignore

from __future__ import annotations

import pytest
from modelbase.ode import Model


def function1(*args):
    pass


def function2(*args):
    pass


DEFAULT_COMPOUNDS = [
    "s11",
    "s12",
    "s13",
    "s21",
    "s22",
    "s23",
    "p11",
    "p12",
    "p13",
    "p21",
    "p22",
    "p23",
    "m11",
    "m12",
    "m13",
    "m21",
    "m22",
    "m23",
]
DEFAULT_PARAMETERS = {"k11": 1, "k12": 2, "k13": 3, "k21": 4, "k22": 5, "k23": 6}
DEFAULT_RATE = dict(
    function=function1,
    substrates=["s11", "s12"],
    products=["p11", "p12"],
    modifiers=["m11", "m12"],
    parameters=["k11", "k12"],
    reversible=False,
)
DEFAULT_MODEL = Model(
    compounds=DEFAULT_COMPOUNDS,
    parameters=DEFAULT_PARAMETERS,
    rates={"rate1": DEFAULT_RATE},
)


def test_default_model():
    model = DEFAULT_MODEL.copy()
    assert model.get_compounds() == [
        "s11",
        "s12",
        "s13",
        "s21",
        "s22",
        "s23",
        "p11",
        "p12",
        "p13",
        "p21",
        "p22",
        "p23",
        "m11",
        "m12",
        "m13",
        "m21",
        "m22",
        "m23",
    ]
    assert model.get_parameters() == {
        "k11": 1,
        "k12": 2,
        "k13": 3,
        "k21": 4,
        "k22": 5,
        "k23": 6,
    }

    rate = model.rates["rate1"]
    assert rate["function"].__name__ == "function1"
    assert rate["substrates"] == ["s11", "s12"]
    assert rate["products"] == ["p11", "p12"]
    assert rate["modifiers"] == ["m11", "m12"]
    assert rate["parameters"] == ["k11", "k12"]
    assert rate["reversible"] is False
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k11", "k12"]


def test_update_nothing():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1")

    rate = model.rates["rate1"]
    assert rate["function"].__name__ == "function1"
    assert rate["substrates"] == ["s11", "s12"]
    assert rate["products"] == ["p11", "p12"]
    assert rate["modifiers"] == ["m11", "m12"]
    assert rate["parameters"] == ["k11", "k12"]
    assert rate["reversible"] is False
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k11", "k12"]


###############################################################################
# Function
###############################################################################


def test_update_function():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", function=function2)

    rate = model.rates["rate1"]
    assert rate["function"].__name__ == "function2"


###############################################################################
# Substrates
###############################################################################


def test_update_substrates():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", substrates=["s21", "s22"])

    rate = model.rates["rate1"]
    assert rate["substrates"] == ["s21", "s22"]
    assert rate["dynamic_variables"] == ["s21", "s22", "m11", "m12"]
    assert rate["args"] == ["s21", "s22", "m11", "m12", "k11", "k12"]


def test_update_substrates_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", substrates=["s21"])

    rate = model.rates["rate1"]
    assert rate["substrates"] == ["s21"]
    assert rate["dynamic_variables"] == ["s21", "m11", "m12"]
    assert rate["args"] == ["s21", "m11", "m12", "k11", "k12"]


def test_update_substrates_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", substrates=["s21", "s22", "s23"])

    rate = model.rates["rate1"]
    assert rate["substrates"] == ["s21", "s22", "s23"]
    assert rate["dynamic_variables"] == ["s21", "s22", "s23", "m11", "m12"]
    assert rate["args"] == ["s21", "s22", "s23", "m11", "m12", "k11", "k12"]


###############################################################################
# Products
###############################################################################


def test_update_products():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", products=["p21", "p22"])

    rate = model.rates["rate1"]
    assert rate["products"] == ["p21", "p22"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k11", "k12"]


def test_update_products_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", products=["p21"])

    rate = model.rates["rate1"]
    assert rate["products"] == ["p21"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k11", "k12"]


def test_update_products_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", products=["p21", "p22", "p23"])

    rate = model.rates["rate1"]
    assert rate["products"] == ["p21", "p22", "p23"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k11", "k12"]


###############################################################################
# Modifiers
###############################################################################


def test_update_modifiers():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", modifiers=["m21", "m22"])

    rate = model.rates["rate1"]
    assert rate["modifiers"] == ["m21", "m22"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m21", "m22"]
    assert rate["args"] == ["s11", "s12", "m21", "m22", "k11", "k12"]


def test_update_modifiers_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", modifiers=["m21"])

    rate = model.rates["rate1"]
    assert rate["modifiers"] == ["m21"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m21"]
    assert rate["args"] == ["s11", "s12", "m21", "k11", "k12"]


def test_update_modifiers_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", modifiers=["m21", "m22", "m23"])

    rate = model.rates["rate1"]
    assert rate["modifiers"] == ["m21", "m22", "m23"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m21", "m22", "m23"]
    assert rate["args"] == ["s11", "s12", "m21", "m22", "m23", "k11", "k12"]


###############################################################################
# Parameters
###############################################################################


def test_update_parameter():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", parameters=["k21", "k22"])

    rate = model.rates["rate1"]
    assert rate["parameters"] == ["k21", "k22"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k21", "k22"]


def test_update_parameter_one_less():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", parameters=["k21"])

    rate = model.rates["rate1"]
    assert rate["parameters"] == ["k21"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k21"]


def test_update_parameter_one_more():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", parameters=["k21", "k22", "k23"])

    rate = model.rates["rate1"]
    assert rate["parameters"] == ["k21", "k22", "k23"]
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k21", "k22", "k23"]


###############################################################################
# Reversible
###############################################################################


def test_update_irreversible_to_reversible():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", reversible=True)

    rate = model.rates["rate1"]
    assert rate["reversible"] is True
    assert rate["dynamic_variables"] == ["s11", "s12", "p11", "p12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "p11", "p12", "m11", "m12", "k11", "k12"]


def test_update_reversible_to_irreversible():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", reversible=True)
    model.update_rate(rate_name="rate1", reversible=False)

    rate = model.rates["rate1"]
    assert rate["reversible"] is False
    assert rate["dynamic_variables"] == ["s11", "s12", "m11", "m12"]
    assert rate["args"] == ["s11", "s12", "m11", "m12", "k11", "k12"]


###############################################################################
# Dynamic variables
###############################################################################


def test_update_dynamic_variables():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", dynamic_variables=["m11", "m12", "s11", "s12"])

    rate = model.rates["rate1"]
    assert rate["dynamic_variables"] == ["m11", "m12", "s11", "s12"]
    assert rate["args"] == ["m11", "m12", "s11", "s12", "k11", "k12"]


def test_update_other_keep_dynamic_variables():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", dynamic_variables=["m11", "m12", "s11", "s12"])
    model.update_rate(rate_name="rate1", function=function2)

    rate = model.rates["rate1"]
    assert rate["function"].__name__ == "function2"
    assert rate["dynamic_variables"] == ["m11", "m12", "s11", "s12"]
    assert rate["args"] == ["m11", "m12", "s11", "s12", "k11", "k12"]


###############################################################################
# Args
###############################################################################


def test_update_args():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", args=["k11", "k12", "m11", "m12", "s11", "s12"])

    rate = model.rates["rate1"]
    assert rate["dynamic_variables"] == ["m11", "m12", "s11", "s12"]
    assert rate["args"] == ["k11", "k12", "m11", "m12", "s11", "s12"]


def test_update_other_keep_args():
    model = DEFAULT_MODEL.copy()
    model.update_rate(rate_name="rate1", args=["k11", "k12", "m11", "m12", "s11", "s12"])
    model.update_rate(rate_name="rate1", function=function2)

    rate = model.rates["rate1"]
    assert rate["function"].__name__ == "function2"
    assert rate["dynamic_variables"] == ["m11", "m12", "s11", "s12"]
    assert rate["args"] == ["k11", "k12", "m11", "m12", "s11", "s12"]
