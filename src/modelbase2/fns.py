"""Module containing functions for reactions and derived quatities."""

from __future__ import annotations

__all__ = ["constant", "diffusion_1s_1p", "div", "mass_action_1s", "mass_action_1s_1p", "mass_action_2s", "mass_action_2s_1p", "michaelis_menten_1s", "michaelis_menten_2s", "michaelis_menten_2s_ping_pong", "michaelis_menten_3s", "michaelis_menten_3s_ping_pong", "minus", "moiety_1s", "moiety_2s", "mul", "neg", "neg_div", "one_div", "proportional", "twice"]

###############################################################################
# General functions
###############################################################################


def constant(x: float) -> float:
    """Constant function."""
    return x


def neg(x: float) -> float:
    """Negation function."""
    return -x


def minus(x: float, y: float) -> float:
    """Subtraction function."""
    return x - y


def mul(x: float, y: float) -> float:
    """Multiplication function."""
    return x * y


def div(x: float, y: float) -> float:
    """Division function."""
    return x / y


def one_div(x: float) -> float:
    """Reciprocal function."""
    return 1.0 / x


def neg_div(x: float, y: float) -> float:
    """Negated division function."""
    return -x / y


def twice(x: float) -> float:
    """Twice function."""
    return x * 2


def proportional(x: float, y: float) -> float:
    """Proportional function."""
    return x * y


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
# Reactions: mass action type
###############################################################################


def mass_action_1s(s1: float, k: float) -> float:
    """Irreversible mass action reaction with one substrate."""
    return k * s1


def mass_action_1s_1p(s1: float, p1: float, kf: float, kr: float) -> float:
    """Reversible mass action reaction with one substrate and one product."""
    return kf * s1 - kr * p1


def mass_action_2s(s1: float, s2: float, k: float) -> float:
    """Irreversible mass action reaction with two substrates."""
    return k * s1 * s2


def mass_action_2s_1p(s1: float, s2: float, p1: float, kf: float, kr: float) -> float:
    """Reversible mass action reaction with two substrates and one product."""
    return kf * s1 * s2 - kr * p1


###############################################################################
# Reactions: michaelis-menten type
###############################################################################


def michaelis_menten_1s(s1: float, vmax: float, km1: float) -> float:
    """Irreversible Michaelis-Menten equation for one substrate."""
    return s1 * vmax / (s1 + km1)


# def michaelis_menten_1s_1i(
#     s: float,
#     i: float,
#     vmax: float,
#     km: float,
#     ki: float,
# ) -> float:
#     """Irreversible Michaelis-Menten equation for one substrate and one inhibitor."""
#     return vmax * s / (s + km * (1 + i / ki))


# def michaelis_menten_1s_1a(
#     s: float,
#     a: float,
#     vmax: float,
#     km: float,
#     ka: float,
# ) -> float:
#     """Irreversible Michaelis-Menten equation for one substrate and one activator."""
#     return vmax * s / (s + km * (1 + ka / a))


def michaelis_menten_2s(
    s1: float,
    s2: float,
    vmax: float,
    km1: float,
    km2: float,
) -> float:
    """Michaelis-Menten equation for two substrates."""
    return vmax * s1 * s2 / ((km1 + s1) * (km2 + s2))


def michaelis_menten_2s_ping_pong(
    s1: float,
    s2: float,
    vmax: float,
    km1: float,
    km2: float,
) -> float:
    """Michaelis-Menten equation (ping-pong) for two substrates."""
    return vmax * s1 * s2 / (s1 * s2 + km1 * s2 + km2 * s1)


def michaelis_menten_3s(
    s1: float,
    s2: float,
    s3: float,
    vmax: float,
    km1: float,
    km2: float,
    km3: float,
) -> float:
    """Michaelis-Menten equation for three substrates."""
    return vmax * s1 * s2 * s3 / ((km1 + s1) * (km2 + s2) * (km3 + s3))


def michaelis_menten_3s_ping_pong(
    s1: float,
    s2: float,
    s3: float,
    vmax: float,
    km1: float,
    km2: float,
    km3: float,
) -> float:
    """Michaelis-Menten equation (ping-pong) for three substrates."""
    return (
        vmax * s1 * s2 * s3 / (s1 * s2 + km1 * s2 * s3 + km2 * s1 * s3 + km3 * s1 * s2)
    )


###############################################################################
# Reactions: michaelis-menten type
###############################################################################


def diffusion_1s_1p(inside: float, outside: float, k: float) -> float:
    """Diffusion reaction with one substrate and one product."""
    return k * (outside - inside)
