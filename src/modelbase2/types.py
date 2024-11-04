from __future__ import annotations

from dataclasses import dataclass

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
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, Self, TypeVar, cast

import numpy as np
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

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


class ModelProtocol(Protocol):
    def get_variable_names(self) -> list[str]: ...
    def get_derived_variable_names(self) -> list[str]: ...
    def get_readout_names(self) -> list[str]: ...
    def get_reaction_names(self) -> list[str]: ...
    def get_parameters(self) -> dict[str, float]: ...
    def update_parameters(self, parameters: dict[str, float]) -> Self: ...

    # User-facing
    def get_args(
        self,
        concs: dict[str, float],
        time: float,
        *,
        include_readouts: bool = True,
    ) -> pd.Series: ...
    def get_full_concs(
        self,
        concs: dict[str, float],
        time: float,
        *,
        include_readouts: bool = True,
    ) -> pd.Series: ...
    def get_fluxes(
        self,
        concs: dict[str, float],
        time: float,
    ) -> pd.Series: ...
    def get_right_hand_side(
        self,
        concs: dict[str, float],
        time: float,
    ) -> pd.Series: ...
    def get_initial_conditions(self) -> ArrayLike: ...

    # For integration
    def _get_rhs(self, /, time: float, concs: Array) -> Array: ...

    # Vectorised
    def _get_args_vectorised(
        self,
        concs: pd.DataFrame,
        *,
        include_readouts: bool,
    ) -> pd.DataFrame: ...
    def _get_fluxes_vectorised(
        self,
        args: pd.DataFrame,
    ) -> pd.DataFrame: ...


class IntegratorProtocol(Protocol):
    """Interface for integrators"""

    def __init__(
        self,
        rhs: Callable,
        y0: ArrayLike,
    ) -> None: ...

    def reset(self) -> None: ...

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]: ...

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> tuple[float | None, ArrayLike | None]: ...


@dataclass(slots=True)
class Derived:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class DerivedVariable:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class DerivedParameter:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class DerivedStoichiometry:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class Reaction:
    fn: DerivedFn
    stoichiometry: dict[str, float | DerivedStoichiometry]
    args: list[str]


@dataclass(slots=True)
class Readout:
    fn: DerivedFn
    args: list[str]
