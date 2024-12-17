from __future__ import annotations

from modelbase.ode import Model
from modelbase.ode.utils import ratefunctions as rf


def v1(s2: float, k: float, k1: float, n: float) -> float:
    return k1 / (1 + (s2 / k) ** n)  # type: ignore


def phase_plane_model() -> Model:
    m = Model()
    m.add_parameters(
        {
            "k1": 20,
            "k2": 5,
            "k3": 5,
            "k4": 5,
            "k5": 2,
            "K": 1,
            "n": 4,
        }
    )
    m.add_compounds(["s1", "s2"])
    m.add_rate(
        rate_name="v1", function=v1, substrates=["s2"], parameters=["K", "k1", "n"]
    )
    m.add_rate(rate_name="v2", function=rf.constant, parameters=["k2"])
    m.add_rate(
        rate_name="v3", function=rf.mass_action_1, substrates=["s1"], parameters=["k3"]
    )
    m.add_rate(
        rate_name="v4", function=rf.mass_action_1, substrates=["s2"], parameters=["k4"]
    )
    m.add_rate(
        rate_name="v5", function=rf.mass_action_1, substrates=["s1"], parameters=["k5"]
    )

    m.add_stoichiometries_by_compounds(
        {
            "s1": {"v1": 1, "v3": -1, "v5": -1},
            "s2": {"v2": 1, "v5": 1, "v4": -1},
        }
    )
    return m


def upper_glycolysis() -> Model:
    # Instantiate model
    m = Model(
        {
            "k1": 0.25,
            "k2": 1,
            "k3": 1,
            "k3m": 1,
            "k4": 1,
            "k5": 1,
            "k6": 1,
            "k7": 2.5,
        }
    )
    m.add_compounds(["GLC", "G6P", "F6P", "FBP", "ATP", "ADP"])

    m.add_reaction(
        rate_name="v1",
        function=rf.constant,
        stoichiometry={"GLC": 1},
        parameters=["k1"],
    )
    m.add_reaction(
        rate_name="v2",
        function=rf.mass_action_2,
        stoichiometry={"GLC": -1, "ATP": -1, "G6P": 1, "ADP": 1},
        parameters=["k2"],
    )
    m.add_reaction(
        rate_name="v3",
        function=rf.reversible_mass_action_1_1,
        stoichiometry={"G6P": -1, "F6P": 1},
        parameters=["k3", "k3m"],
        reversible=True,
    )
    m.add_reaction(
        rate_name="v4",
        function=rf.mass_action_2,
        stoichiometry={"F6P": -1, "ATP": -1, "ADP": 1, "FBP": 1},
        parameters=["k4"],
    )
    m.add_reaction(
        rate_name="v5",
        function=rf.mass_action_1,
        stoichiometry={"FBP": -1, "F6P": 1},
        parameters=["k5"],
    )
    m.add_reaction(
        rate_name="v6",
        function=rf.mass_action_1,
        stoichiometry={"FBP": -1},
        parameters=["k6"],
    )
    m.add_reaction(
        rate_name="v7",
        function=rf.mass_action_1,
        stoichiometry={"ADP": -1, "ATP": 1},
        parameters=["k7"],
    )

    return m
