"""Test call function of fn_to_sympy."""

import sympy

import mxlpy
from mxlpy import fns
from mxlpy.fns import mass_action_1s
from mxlpy.meta.source_tools import fn_to_sympy
from mxlpy.types import Float


def using_inner_l1(x: Float, y: Float) -> Float:
    return mass_action_1s(x, y) + y


def using_inner_l2(x: Float, y: Float) -> Float:
    return fns.mass_action_1s(x, y) + y


def using_inner_l3(x: Float, y: Float) -> Float:
    return mxlpy.fns.mass_action_1s(x, y) + y


def test_call_level1() -> None:
    assert (
        sympy.latex(fn_to_sympy(using_inner_l1, model_args=sympy.symbols("x y")))
        == "x y + y"
    )


def test_call_level2() -> None:
    assert (
        sympy.latex(fn_to_sympy(using_inner_l2, model_args=sympy.symbols("x y")))
        == "x y + y"
    )


def test_call_level3() -> None:
    assert (
        sympy.latex(fn_to_sympy(using_inner_l3, model_args=sympy.symbols("x y")))
        == "x y + y"
    )
