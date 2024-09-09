# type: ignore

from __future__ import annotations

import pytest

from modelbase2.ode import Model


def test_generate_function_source_code(multiline_comparison):
    def func(x):
        return x

    model = Model()
    model.add_function("func", func)
    model.add_function("func2", lambda x: x)
    multiline_comparison(
        [
            "def func(x):",
            "    return x",
            "def func2(x):",
            "    return",
        ],
        model._generate_function_source_code(),
    )
