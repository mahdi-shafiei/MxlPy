"""Test call function of model_fn_to_sympy."""

import sympy

import modelbase2
from modelbase2 import fns
from modelbase2.experimental.symbolic import model_fn_to_sympy
from modelbase2.fns import mass_action_1s
from modelbase2.types import Float


def using_inner_l1(x: Float, y: Float) -> Float:
    return mass_action_1s(x, y) + y


def using_inner_l2(x: Float, y: Float) -> Float:
    return fns.mass_action_1s(x, y) + y


def using_inner_l3(x: Float, y: Float) -> Float:
    return modelbase2.fns.mass_action_1s(x, y) + y


def test_call_level1() -> None:
    assert (
        sympy.latex(model_fn_to_sympy(using_inner_l1, model_args=sympy.symbols("x y")))
        == "x y + y"
    )


def test_call_level2() -> None:
    assert (
        sympy.latex(model_fn_to_sympy(using_inner_l2, model_args=sympy.symbols("x y")))
        == "x y + y"
    )


def test_call_level3() -> None:
    assert (
        sympy.latex(model_fn_to_sympy(using_inner_l3, model_args=sympy.symbols("x y")))
        == "x y + y"
    )
