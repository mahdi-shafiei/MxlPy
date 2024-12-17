# type: ignore

from __future__ import annotations

import pytest
from modelbase.ode import Model


def test_generate_rates_source_code(multiline_comparison):
    model = Model()
    model.add_rate(
        rate_name="v1",
        function=lambda x, y, ATP, ADP: x * ATP - y * ADP,
        substrates=["x"],
        products=["y"],
        modifiers=["ATP", "ADP"],
        parameters=["k1"],
        reversible=True,
        **{"common_name": "rate 1"},
    )
    rate_functions, rates = model._generate_rates_source_code(include_meta_info=False)
    multiline_comparison(
        ["def v1(x, y, ATP, ADP):", "    return x * ATP - y * ADP"], rate_functions
    )
    multiline_comparison(
        [
            "m.add_rate(",
            "    rate_name='v1',",
            "    function=v1,",
            "    substrates=['x'],",
            "    products=['y'],",
            "    modifiers=['ATP', 'ADP'],",
            "    parameters=['k1'],",
            "    reversible=True,",
            "    args=['x', 'y', 'ATP', 'ADP', 'k1'],",
            ")",
        ],
        rates,
    )

    rate_functions, rates = model._generate_rates_source_code(include_meta_info=True)
    multiline_comparison(
        [
            "m.add_rate(",
            "    rate_name='v1',",
            "    function=v1,",
            "    substrates=['x'],",
            "    products=['y'],",
            "    modifiers=['ATP', 'ADP'],",
            "    parameters=['k1'],",
            "    reversible=True,",
            "    args=['x', 'y', 'ATP', 'ADP', 'k1'],",
            "    **{'common_name': 'rate 1'}",
            ")",
        ],
        rates,
    )
