from __future__ import annotations

__all__ = [
    "Any",
    "Axis",
    "Callable",
    "Figure",
    "Protocol",
    "cast",
]

# Re-exporting some types here, because their imports have
# changed between Python versions and I have no interest in
# fixing it in every file
from collections.abc import Callable, Hashable, Iterable
from typing import Any, ParamSpec, Protocol, TypeVar, cast

import numpy as np
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure
from numpy.typing import NDArray

type DerivedFn = Callable[..., float]
type Array = NDArray[np.float64]
type Number = float | list[float] | Array

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

Axes = NDArray[Axis]  # type: ignore
ArrayLike = NDArray[np.float64] | list[float]

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
