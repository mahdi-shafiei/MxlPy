from __future__ import annotations


def constant(x: float) -> float:
    return x


def mass_action_1s(x: float, k: float) -> float:
    return k * x
