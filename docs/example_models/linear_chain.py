from __future__ import annotations

from modelbase2 import Model

from .shared import constant, mass_action_1s


def get_linear_chain_2v() -> Model:
    return (
        Model()
        .add_variables({"x": 1.0, "y": 1.0})
        .add_parameters({"k1": 1.0, "k2": 2.0, "k3": 1.0})
        .add_reaction("v1", constant, stoichiometry={"x": 1}, args=["k1"])
        .add_reaction(
            "v2",
            mass_action_1s,
            stoichiometry={"x": -1, "y": 1},
            args=["k2", "x"],
        )
        .add_reaction(
            "v3",
            mass_action_1s,
            stoichiometry={"y": -1},
            args=["k3", "y"],
        )
    )
