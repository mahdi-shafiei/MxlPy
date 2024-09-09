from __future__ import annotations

from modelbase2.ode import DerivedStoichiometry, Model


def return_arg(x: float) -> float:
    return x


def return_minus_arg(x: float) -> float:
    return -x


def test_add() -> None:
    m = Model()
    m.add_compounds(["x", "y"])
    m.add_parameters({"p1": 1, "p2": 2})
    m.add_reaction_from_args(
        rate_name="v1",
        function=return_arg,
        stoichiometry={"y": 2},
        derived_stoichiometry={
            "x": DerivedStoichiometry(return_minus_arg, args=["p1"]),
        },
        args=["x"],
    )

    assert m.stoichiometries == {
        "v1": {
            "x": -1,
            "y": 2,
        }
    }
    assert m.stoichiometries_by_compounds == {
        "x": {"v1": -1},
        "y": {"v1": 2},
    }


def test_update_parameter() -> None:
    m = Model()
    m.add_compounds(["x", "y"])
    m.add_parameters({"p1": 1, "p2": 2})
    m.add_reaction_from_args(
        rate_name="v1",
        function=return_arg,
        stoichiometry={"y": 2},
        derived_stoichiometry={
            "x": DerivedStoichiometry(return_minus_arg, args=["p1"]),
        },
        args=["x"],
    )
    m.update_parameter("p1", 2)

    assert m.stoichiometries == {
        "v1": {
            "x": -2,
            "y": 2,
        }
    }
    assert m.stoichiometries_by_compounds == {
        "x": {"v1": -2},
        "y": {"v1": 2},
    }
