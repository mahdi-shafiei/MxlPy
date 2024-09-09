# type: ignore

from __future__ import annotations

import unittest

import pytest

from modelbase2.core.utils import (
    RE_LAMBDA_ALGEBRAIC_MODULE_FUNC,
    RE_LAMBDA_FUNC,
    RE_LAMBDA_RATE_FUNC,
    convert_id_to_sbml,
    convert_sbml_id,
    functionify_lambda,
    get_formatted_function_source_code,
    get_function_source_code,
)


def test_get_function_source_code(multiline_comparison):
    def func(a):
        return a

    multiline_comparison(
        ["    def func(a):", "        return a"], get_function_source_code(func)
    )


def test_get_function_source_code_exec(multiline_comparison):
    exec("def func(a):\n    return a")  # nosec
    function = locals()["func"]
    function.__source__ = "def func(a):\n    return a"

    multiline_comparison(
        ["def func(a):", "    return a"],
        get_function_source_code(function),
    )


def test_get_formatted_function_source_code_fail_on_wrong_type():
    with pytest.raises(ValueError):
        get_formatted_function_source_code(
            function_name="v1", function=lambda *args: 0, function_type="wurst"
        )


def test_get_formatted_function_source_code(multiline_comparison):
    def func(a):
        return a

    multiline_comparison(
        ("def func(a):", "    return a"),
        get_formatted_function_source_code(
            function_name="func", function=func, function_type="rate"
        ),
    )


def test_get_formatted_function_source_code_exec(multiline_comparison):
    exec("def func(a):\n    return a")  # nosec
    function = locals()["func"]
    function.__source__ = "def func(a):\n    return a"

    multiline_comparison(
        ("def func(a):", "    return a"),
        get_formatted_function_source_code(
            function_name="func", function=function, function_type="rate"
        ),
    )


def test_no_compounds(multiline_comparison):
    code = "lambda k_in: k_in,"
    expected = ("def v1(k_in):", "    return k_in")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code, function_name="v1", pattern=RE_LAMBDA_RATE_FUNC
        ),
    )


def test_no_compounds_named_arg(multiline_comparison):
    code = "function=lambda k_in: k_in,"
    expected = ("def v1(k_in):", "    return k_in")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code, function_name="v1", pattern=RE_LAMBDA_RATE_FUNC
        ),
    )


def test_one_compound_one_parameter(multiline_comparison):
    code = "function=lambda x, kf: kf * x,"
    expected = ("def v1(x, kf):", "    return kf * x")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code, function_name="v1", pattern=RE_LAMBDA_RATE_FUNC
        ),
    )


def test_two_compounds_two_parameters(multiline_comparison):
    code = "function=lambda x, y, kf, kr: (kf * x) - (kr * y),"
    expected = ("def v1(x, y, kf, kr):", "    return (kf * x) - (kr * y)")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code, function_name="v1", pattern=RE_LAMBDA_RATE_FUNC
        ),
    )


def test_single_compound(multiline_comparison):
    code = "lambda x: (x, ),"
    expected = ("def mod1(x):", "    return (x, )")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code,
            function_name="mod1",
            pattern=RE_LAMBDA_ALGEBRAIC_MODULE_FUNC,
        ),
    )


def test_single_compoundd_arg(multiline_comparison):
    code = "function=lambda x: (x, ),"
    expected = ("def mod1(x):", "    return (x, )")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code,
            function_name="mod1",
            pattern=RE_LAMBDA_ALGEBRAIC_MODULE_FUNC,
        ),
    )


def test_two_compounds(multiline_comparison):
    code = "function=lambda x, y, keq: (keq * (x + y), keq * (x + y)),"
    expected = ("def mod1(x, y, keq):", "    return (keq * (x + y), keq * (x + y))")
    multiline_comparison(
        expected,
        functionify_lambda(
            lambda_function_code=code,
            function_name="mod1",
            pattern=RE_LAMBDA_ALGEBRAIC_MODULE_FUNC,
        ),
    )


class SBMLIdTests(unittest.TestCase):
    def test_sbml_id_handling(self):
        id_ = "cpd-1"
        sbml_id = convert_id_to_sbml(id_=id_, prefix="CPD")
        self.assertEqual(sbml_id, "cpd__45__1")
        self.assertEqual(id_, convert_sbml_id(sbml_id=sbml_id, prefix="CPD"))

    def test_sbml_id_handling_nonalpha_start(self):
        id_ = "1"
        sbml_id = convert_id_to_sbml(id_=id_, prefix="CPD")
        self.assertEqual(sbml_id, "CPD_1")
        self.assertEqual(id_, convert_sbml_id(sbml_id=sbml_id, prefix="CPD"))

    def test_sbml_id_handling_non_ascii_start(self):
        id_ = "-"
        sbml_id = convert_id_to_sbml(id_=id_, prefix="CPD")
        self.assertEqual(sbml_id, "CPD___45__")
        self.assertEqual(id_, convert_sbml_id(sbml_id=sbml_id, prefix="CPD"))
