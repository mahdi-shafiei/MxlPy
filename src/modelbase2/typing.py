from __future__ import annotations

__all__ = [
    "Figure",
    "Axis",
]

from collections.abc import Hashable, Iterable
from typing import TypeVar, Union

import numpy as np
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure
from numpy.typing import NDArray

type Array = NDArray[np.float64]
Number = Union[
    float,
    list[float],
    Array,
]

Axes = NDArray[Axis]  # type: ignore
ArrayLike = Union[NDArray[np.float64], list[float]]

T = TypeVar("T")
V = TypeVar("V")
Tin = TypeVar("Tin")
Tout = TypeVar("Tout")
Ti = TypeVar("Ti", bound=Iterable)
K = TypeVar("K", bound=Hashable)


def unwrap(x: T | None) -> T:
    if x is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return x


def default_if_none(el: T | None, default: T) -> T:
    return default if el is None else el
