"""Collection of shared functions for the example models."""

from __future__ import annotations


def constant(x: float) -> float:
    """Constant function."""
    return x


def mass_action_1s(x: float, k: float) -> float:
    """Mass action reaction with one substrate."""
    return k * x


def mass_action_1s_1p(x: float, y: float, kf: float, kr: float) -> float:
    """Mass action reaction with one substrate and one product."""
    return kf * x - kr * y


def mass_action_2s(x: float, y: float, k: float) -> float:
    """Mass action reaction with two substrates."""
    return k * x * y


def michaelis_menten_1s(s: float, vmax: float, km: float) -> float:
    """Irreversible Michaelis-Menten equation for one substrate."""
    return s * vmax / (s + km)
