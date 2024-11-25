from __future__ import annotations


def constant(x: float) -> float:
    return x


def mass_action_1s(x: float, k: float) -> float:
    return k * x


def mass_action_1s_1p(x: float, y: float, kf: float, kr: float) -> float:
    return kf * x - kr * y


def mass_action_2s(x: float, y: float, k: float) -> float:
    return k * x * y
