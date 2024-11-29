"""Module containing functions for reactions and derived quatities."""

from __future__ import annotations

###############################################################################
# General functions
###############################################################################


def constant(x: float) -> float:
    """Constant function."""
    return x


###############################################################################
# Derived functions
###############################################################################


def moiety_1s(
    x: float,
    x_total: float,
) -> float:
    """General moiety for one substrate."""
    return x_total - x


def moiety_2s(
    x1: float,
    x2: float,
    x_total: float,
) -> float:
    """General moiety for two substrates."""
    return x_total - x1 - x2


###############################################################################
# Reactions
###############################################################################


def mass_action_1s(s1: float, k: float) -> float:
    """Mass action reaction with one substrate."""
    return k * s1


def mass_action_1s_1p(s1: float, p1: float, kf: float, kr: float) -> float:
    """Mass action reaction with one substrate and one product."""
    return kf * s1 - kr * p1


def mass_action_2s(s1: float, s2: float, k: float) -> float:
    """Mass action reaction with two substrates."""
    return k * s1 * s2


def mass_action_2s_1p(s1: float, s2: float, p1: float, kf: float, kr: float) -> float:
    """Mass action reaction with two substrates and one product."""
    return kf * s1 * s2 - kr * p1


def michaelis_menten_1s(s: float, vmax: float, km: float) -> float:
    """Irreversible Michaelis-Menten equation for one substrate."""
    return s * vmax / (s + km)


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
