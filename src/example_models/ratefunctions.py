"""Collection of rate functions for enzyme kinetics."""

from __future__ import annotations

__all__ = [
    "competitive_activation",
    "competitive_inhibition",
    "constant",
    "hill",
    "mass_action_1",
    "mass_action_2",
    "mass_action_3",
    "mass_action_4",
    "mass_action_variable",
    "michaelis_menten",
    "mixed_activation",
    "mixed_inhibition",
    "noncompetitive_activation",
    "noncompetitive_inhibition",
    "ordered_2",
    "ordered_2_2",
    "ping_pong_2",
    "ping_pong_3",
    "ping_pong_4",
    "random_order_2",
    "random_order_2_2",
    "reversible_mass_action_1_1",
    "reversible_mass_action_1_2",
    "reversible_mass_action_1_3",
    "reversible_mass_action_1_4",
    "reversible_mass_action_2_1",
    "reversible_mass_action_2_2",
    "reversible_mass_action_2_3",
    "reversible_mass_action_2_4",
    "reversible_mass_action_3_1",
    "reversible_mass_action_3_2",
    "reversible_mass_action_3_3",
    "reversible_mass_action_3_4",
    "reversible_mass_action_4_1",
    "reversible_mass_action_4_2",
    "reversible_mass_action_4_3",
    "reversible_mass_action_4_4",
    "reversible_michaelis_menten",
    "reversible_michaelis_menten_keq",
    "reversible_noncompetitive_inhibition",
    "reversible_noncompetitive_inhibition_keq",
    "reversible_uncompetitive_inhibition",
    "reversible_uncompetitive_inhibition_keq",
    "uncompetitive_activation",
    "uncompetitive_inhibition",
]


from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mxlpy.types import Float


def constant(k: Float) -> Float:
    """Constant function."""
    return k


###############################################################################
# Mass action
###############################################################################


def mass_action_1(s1: Float, k_fwd: Float) -> Float:
    """Mass action equation for one substrate."""
    return k_fwd * s1


def mass_action_2(s1: Float, s2: Float, k_fwd: Float) -> Float:
    """Mass action equation for two substrates."""
    return k_fwd * s1 * s2


def mass_action_3(s1: Float, s2: Float, s3: Float, k_fwd: Float) -> Float:
    """Mass action equation for three substrates."""
    return k_fwd * s1 * s2 * s3


def mass_action_4(s1: Float, s2: Float, s3: Float, s4: Float, k_fwd: Float) -> Float:
    """Mass action equation for four substrates."""
    return k_fwd * s1 * s2 * s3 * s4


def mass_action_variable(*args: Float) -> Float:
    """Mass action equation for variable number of substrates."""
    return reduce(mul, args, 1)


###############################################################################
# Reversible Mass action
###############################################################################


def reversible_mass_action_1_1(
    s1: Float,
    p1: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for one substrate and one product."""
    return k_fwd * s1 - k_bwd * p1


def reversible_mass_action_2_1(
    s1: Float,
    s2: Float,
    p1: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for two substrates and one product."""
    return k_fwd * s1 * s2 - k_bwd * p1


def reversible_mass_action_3_1(
    s1: Float,
    s2: Float,
    s3: Float,
    p1: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for three substrates and one product."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1


def reversible_mass_action_4_1(
    s1: Float,
    s2: Float,
    s3: Float,
    s4: Float,
    p1: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for four substrates and one product."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1


def reversible_mass_action_1_2(
    s1: Float,
    p1: Float,
    p2: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for one substrate and two products."""
    return k_fwd * s1 - k_bwd * p1 * p2


def reversible_mass_action_2_2(
    s1: Float,
    s2: Float,
    p1: Float,
    p2: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for two substrates and two products."""
    return k_fwd * s1 * s2 - k_bwd * p1 * p2


def reversible_mass_action_3_2(
    s1: Float,
    s2: Float,
    s3: Float,
    p1: Float,
    p2: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for three substrates and two products."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1 * p2


def reversible_mass_action_4_2(
    s1: Float,
    s2: Float,
    s3: Float,
    s4: Float,
    p1: Float,
    p2: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for four substrates and two products."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1 * p2


def reversible_mass_action_1_3(
    s1: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for one substrate and three products."""
    return k_fwd * s1 - k_bwd * p1 * p2 * p3


def reversible_mass_action_2_3(
    s1: Float,
    s2: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for two substrates and three products."""
    return k_fwd * s1 * s2 - k_bwd * p1 * p2 * p3


def reversible_mass_action_3_3(
    s1: Float,
    s2: Float,
    s3: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for three substrates and three products."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1 * p2 * p3


def reversible_mass_action_4_3(
    s1: Float,
    s2: Float,
    s3: Float,
    s4: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for four substrates and three products."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1 * p2 * p3


def reversible_mass_action_1_4(
    s1: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    p4: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for one substrate and four products."""
    return k_fwd * s1 - k_bwd * p1 * p2 * p3 * p4


def reversible_mass_action_2_4(
    s1: Float,
    s2: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    p4: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for two substrates and four products."""
    return k_fwd * s1 * s2 - k_bwd * p1 * p2 * p3 * p4


def reversible_mass_action_3_4(
    s1: Float,
    s2: Float,
    s3: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    p4: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for three substrates and four products."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1 * p2 * p3 * p4


def reversible_mass_action_4_4(
    s1: Float,
    s2: Float,
    s3: Float,
    s4: Float,
    p1: Float,
    p2: Float,
    p3: Float,
    p4: Float,
    k_fwd: Float,
    k_bwd: Float,
) -> Float:
    """Reversible mass action equation for four substrates and four products."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1 * p2 * p3 * p4


###############################################################################
# Michaelis Menten
###############################################################################


def michaelis_menten(s: Float, vmax: Float, km: Float) -> Float:
    """Irreversible Michaelis-Menten equation for one substrate."""
    return s * vmax / (s + km)


def competitive_inhibition(
    s: Float, i: Float, vmax: Float, km: Float, ki: Float
) -> Float:
    """Competitive inhibition."""
    return vmax * s / (s + km * (1 + i / ki))


def competitive_activation(
    s: Float, a: Float, vmax: Float, km: Float, ka: Float
) -> Float:
    """Competitive activation."""
    return vmax * s / (s + km * (1 + ka / a))


def uncompetitive_inhibition(
    s: Float, i: Float, vmax: Float, km: Float, ki: Float
) -> Float:
    """Uncompetitive inhibition."""
    return vmax * s / (s * (1 + i / ki) + km)


def uncompetitive_activation(
    s: Float, a: Float, vmax: Float, km: Float, ka: Float
) -> Float:
    """Uncompetitive activation."""
    return vmax * s / (s * (1 + ka / a) + km)


def noncompetitive_inhibition(
    s: Float, i: Float, vmax: Float, km: Float, ki: Float
) -> Float:
    """Noncompetitive inhibition."""
    return vmax * s / ((s + km) * (1 + i / ki))


def noncompetitive_activation(
    s: Float, a: Float, vmax: Float, km: Float, ka: Float
) -> Float:
    """Noncompetitive activation."""
    return vmax * s / ((s + km) * (1 + ka / a))


def mixed_inhibition(s: Float, i: Float, vmax: Float, km: Float, ki: Float) -> Float:
    """Mixed inhibition."""
    return vmax * s / (s * (1 + i / ki) + km * (1 + i / ki))


def mixed_activation(s: Float, a: Float, vmax: Float, km: Float, ka: Float) -> Float:
    """Mixed activation."""
    return vmax * s / (s * (1 + ka / a) + km * (1 + ka / a))


###############################################################################
# Reversible Michaelis-Menten
###############################################################################


def reversible_michaelis_menten(
    s: Float,
    p: Float,
    vmax_fwd: Float,
    vmax_bwd: Float,
    kms: Float,
    kmp: Float,
) -> Float:
    """Reversible Michaelis-Menten equation for one substrate and one product."""
    return (vmax_fwd * s / kms - vmax_bwd * p / kmp) / (1 + s / kms + p / kmp)


def reversible_uncompetitive_inhibition(
    s: Float,
    p: Float,
    i: Float,
    vmax_fwd: Float,
    vmax_bwd: Float,
    kms: Float,
    kmp: Float,
    ki: Float,
) -> Float:
    """Reversible uncompetitive inhibition."""
    return (vmax_fwd * s / kms - vmax_bwd * p / kmp) / (
        1 + (s / kms) + (p / kmp) * (1 + i / ki)
    )


def reversible_noncompetitive_inhibition(
    s: Float,
    p: Float,
    i: Float,
    vmax_fwd: Float,
    vmax_bwd: Float,
    kms: Float,
    kmp: Float,
    ki: Float,
) -> Float:
    """Reversible noncompetitive inhibition."""
    return (vmax_fwd * s / kms - vmax_bwd * p / kmp) / (
        (1 + s / kms + p / kmp) * (1 + i / ki)
    )


def reversible_michaelis_menten_keq(
    s: Float,
    p: Float,
    vmax_fwd: Float,
    kms: Float,
    kmp: Float,
    keq: Float,
) -> Float:
    """Reversible Michaelis-Menten equation for one substrate and one product."""
    return vmax_fwd / kms * (s - p / keq) / (1 + s / kms + p / kmp)


def reversible_uncompetitive_inhibition_keq(
    s: Float,
    p: Float,
    i: Float,
    vmax_fwd: Float,
    kms: Float,
    kmp: Float,
    ki: Float,
    keq: Float,
) -> Float:
    """Reversible uncompetitive inhibition."""
    return vmax_fwd / kms * (s - p / keq) / (1 + (s / kms) + (p / kmp) * (1 + i / ki))


def reversible_noncompetitive_inhibition_keq(
    s: Float,
    p: Float,
    i: Float,
    vmax_fwd: Float,
    kms: Float,
    kmp: Float,
    ki: Float,
    keq: Float,
) -> Float:
    """Reversible noncompetitive inhibition."""
    return vmax_fwd / kms * (s - p / keq) / ((1 + s / kms + p / kmp) * (1 + i / ki))


###############################################################################
# Multi-substrate
###############################################################################


def ordered_2(
    a: Float,
    b: Float,
    vmax: Float,
    kma: Float,
    kmb: Float,
    kia: Float,
) -> Float:
    """Ordered Bi Bi reaction."""
    return vmax * a * b / (a * b + kmb * a + kma * b + kia * kmb)


def ordered_2_2(
    a: Float,
    b: Float,
    p: Float,
    q: Float,
    vmaxf: Float,
    vmaxr: Float,
    kma: Float,
    kmb: Float,
    kmp: Float,
    kmq: Float,
    kia: Float,
    kib: Float,
    kip: Float,
    kiq: Float,
) -> Float:
    """Ordered Bi Bi reaction."""
    nominator = vmaxf * a * b / (kia * kmb) - vmaxr * p * q / (kmp * kiq)
    denominator = (
        1
        + (a / kia)
        + (kma * b / (kia * kmb))
        + (kmq * p / (kmp * kiq))
        + (q / kiq)
        + (a * b / (kia * kmb))
        + (kmq * a * p / (kia * kmp * kiq))
        + (kma * b * q / (kia * kmb * kiq))
        + (p * q / (kmp * kiq))
        + (a * b * p / (kia * kmb * kip))
        + (b * p * q) / (kib * kmp * kiq)
    )
    return nominator / denominator


def random_order_2(
    a: Float,
    b: Float,
    vmax: Float,
    kma: Float,
    kmb: Float,
    kia: Float,
) -> Float:
    """Random-order reaction with two substrates."""
    return vmax * a * b / (a * b + kmb * a + kma * b + kia * kmb)


def random_order_2_2(
    a: Float,
    b: Float,
    p: Float,
    q: Float,
    vmaxf: Float,
    vmaxr: Float,
    kmb: Float,
    kmp: Float,
    kia: Float,
    kib: Float,
    kip: Float,
    kiq: Float,
) -> Float:
    """Random-order reaction with two substrates and two products."""
    nominator = vmaxf * a * b / (kia * kmb) - vmaxr * p * q / (kmp * kiq)
    denominator = (
        1
        + (a / kia)
        + (b / kib)
        + (p / kip)
        + (q / kiq)
        + (a * b / (kia * kmb))
        + (p * q / (kmp * kiq))
    )
    return nominator / denominator


def ping_pong_2(
    a: Float,
    b: Float,
    vmax: Float,
    kma: Float,
    kmb: Float,
) -> Float:
    """Ping-pong reaction with two substrates."""
    return vmax * a * b / (a * b + kma * b + kmb * a)


def ping_pong_3(
    a: Float,
    b: Float,
    c: Float,
    vmax: Float,
    kma: Float,
    kmb: Float,
    kmc: Float,
) -> Float:
    """Ping-pong reaction with three substrates."""
    return (vmax * a * b * c) / (
        a * b * c + (kma * b * c) + (kmb * a * c) + (kmc * a * b)
    )


def ping_pong_4(
    a: Float,
    b: Float,
    c: Float,
    d: Float,
    vmax: Float,
    kma: Float,
    kmb: Float,
    kmc: Float,
    kmd: Float,
) -> Float:
    """Ping-pong reaction with four substrates."""
    return (vmax * a * b * c * d) / (
        a * b * c * d
        + (kma * b * c * d)
        + (kmb * a * c * d)
        + (kmc * a * b * d)
        + (kmd * a * b * c)
    )


###############################################################################
# cooperativity
###############################################################################


def hill(s: Float, vmax: Float, kd: Float, n: Float) -> Float:
    """Hill equation."""
    return vmax * s**n / (kd + s**n)


###############################################################################
# Generalised
###############################################################################

# def hanekom()-> float:
#     pass

# def convenience()-> float:
#     pass
