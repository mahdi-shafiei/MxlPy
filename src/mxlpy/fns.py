"""Module containing functions for reactions and derived quatities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mxlpy.types import Float

__all__ = [
    "add",
    "constant",
    "diffusion_1s_1p",
    "div",
    "mass_action_1s",
    "mass_action_1s_1p",
    "mass_action_2s",
    "mass_action_2s_1p",
    "michaelis_menten_1s",
    "michaelis_menten_2s",
    "michaelis_menten_3s",
    "minus",
    "moiety_1s",
    "moiety_2s",
    "mul",
    "neg",
    "neg_div",
    "one_div",
    "proportional",
    "twice",
]


###############################################################################
# General functions
###############################################################################


def constant(x: Float) -> Float:
    """Constant function."""
    return x


def neg(x: Float) -> Float:
    """Negation function."""
    return -x


def minus(x: Float, y: Float) -> Float:
    """Subtraction function."""
    return x - y


def mul(x: Float, y: Float) -> Float:
    """Multiplication function."""
    return x * y


def div(x: Float, y: Float) -> Float:
    """Division function."""
    return x / y


def one_div(x: Float) -> Float:
    """Reciprocal function."""
    return 1.0 / x


def neg_div(x: Float, y: Float) -> Float:
    """Negated division function."""
    return -x / y


def twice(x: Float) -> Float:
    """Twice function."""
    return x * 2


def add(x: Float, y: Float) -> Float:
    """Proportional function."""
    return x + y


def proportional(x: Float, y: Float) -> Float:
    """Proportional function."""
    return x * y


###############################################################################
# Derived functions
###############################################################################


def moiety_1s(
    x: Float,
    x_total: Float,
) -> Float:
    """General moiety for one substrate."""
    return x_total - x


def moiety_2s(
    x1: Float,
    x2: Float,
    x_total: Float,
) -> Float:
    """General moiety for two substrates."""
    return x_total - x1 - x2


###############################################################################
# Reactions: mass action type
###############################################################################


def mass_action_1s(s1: Float, k: Float) -> Float:
    """Irreversible mass action reaction with one substrate."""
    return k * s1


def mass_action_1s_1p(s1: Float, p1: Float, kf: Float, kr: Float) -> Float:
    """Reversible mass action reaction with one substrate and one product."""
    return kf * s1 - kr * p1


def mass_action_2s(s1: Float, s2: Float, k: Float) -> Float:
    """Irreversible mass action reaction with two substrates."""
    return k * s1 * s2


def mass_action_2s_1p(s1: Float, s2: Float, p1: Float, kf: Float, kr: Float) -> Float:
    """Reversible mass action reaction with two substrates and one product."""
    return kf * s1 * s2 - kr * p1


###############################################################################
# Reactions: michaelis-menten type
# For multi-molecular reactions use ping-pong kinetics as default
###############################################################################


def michaelis_menten_1s(s1: Float, vmax: Float, km1: Float) -> Float:
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


def michaelis_menten_2s(
    s1: Float,
    s2: Float,
    vmax: Float,
    km1: Float,
    km2: Float,
) -> Float:
    """Michaelis-Menten equation (ping-pong) for two substrates."""
    return vmax * s1 * s2 / (s1 * s2 + km1 * s2 + km2 * s1)


def michaelis_menten_3s(
    s1: Float,
    s2: Float,
    s3: Float,
    vmax: Float,
    km1: Float,
    km2: Float,
    km3: Float,
) -> Float:
    """Michaelis-Menten equation (ping-pong) for three substrates."""
    return (
        vmax * s1 * s2 * s3 / (s1 * s2 + km1 * s2 * s3 + km2 * s1 * s3 + km3 * s1 * s2)
    )


###############################################################################
# Reactions: michaelis-menten type
###############################################################################


def diffusion_1s_1p(inside: Float, outside: Float, k: Float) -> Float:
    """Diffusion reaction with one substrate and one product."""
    return k * (outside - inside)
