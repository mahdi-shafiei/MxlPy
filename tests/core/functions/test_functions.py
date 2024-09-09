# type: ignore

from __future__ import annotations

import pytest

from modelbase2.ode import Model


# New tests using pytest
def test_init_functions():
    def func1():
        return 1

    def func2():
        return 2

    functions = {
        "func1": func1,
        "func2": func2,
    }
    model = Model(functions=functions)
    assert set(model.functions.keys()) == {"func1", "func2"}


def test_add_function():
    def func(x):
        return x

    model = Model()
    model.add_function("func", func)
    assert model.functions["func"] == func


def test_add_function_lambda():
    model = Model()
    model.add_function("func", lambda x: x)
    assert model.functions["func"].__name__ == "func"
