"""Example model 1."""

from __future__ import annotations

from modelbase2 import Model


def filter_stoichiometry(
    model: Model,
    stoichiometry: dict[str, float],
) -> dict[str, float]:
    """Only use components that are actually compounds in the model.

    Args:
        model: Metabolic model instance
        stoichiometry: Stoichiometry dictionary {component: value}

    """
    new: dict[str, float] = {}
    ids = model.ids
    variables = model.variables
    for k, v in stoichiometry.items():
        if k in variables:
            new[k] = v
        elif k not in ids:
            msg = f"Missing component {k}"
            raise KeyError(msg)
    return new


def constant(x: float) -> float:
    """Constant function."""
    return x


def michaelis_menten_2s(
    s1: float,
    s2: float,
    vmax: float,
    km1: float,
    km2: float,
    ki1: float,
) -> float:
    """Michaelis-Menten equation for two substrates."""
    return vmax * s1 * s2 / (ki1 * km2 + km2 * s1 + km1 * s2 + s1 * s2)


def get_example1() -> Model:
    """Example model 1."""
    model = Model()
    model.add_variables({"x2": 0.0, "x3": 0.0})
    model.add_parameters(
        {
            # These need to be static in order to train the model later
            "x1": 1.0,
            "ATP": 1.0,
            "NADPH": 1.0,
            # v2
            "vmax_v2": 2.0,
            "km_v2_1": 0.1,
            "km_v2_2": 0.1,
            "ki_v2": 0.1,
            # v3
            "vmax_v3": 2.0,
            "km_v3_1": 0.2,
            "km_v3_2": 0.2,
            "ki_v3": 0.2,
        }
    )

    model.add_reaction(
        "v2",
        michaelis_menten_2s,
        filter_stoichiometry(model, {"x1": -1, "ATP": -1, "x2": 1}),
        ["x1", "ATP", "vmax_v2", "km_v2_1", "km_v2_2", "ki_v2"],
    )
    model.add_reaction(
        "v3",
        michaelis_menten_2s,
        filter_stoichiometry(model, {"x1": -1, "NADPH": -1, "x3": 1}),
        ["x1", "ATP", "vmax_v3", "km_v3_1", "km_v3_2", "ki_v3"],
    )
    model.add_reaction("x2_out", constant, {"x2": -1}, ["x2"])
    model.add_reaction("x3_out", constant, {"x3": -1}, ["x3"])

    return model
