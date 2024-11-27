"""Types Module.

This module provides type definitions and utility types for use throughout the project.
It includes type aliases for arrays, numbers, and callable functions, as well as re-exports
of common types from standard libraries.

Classes:
    DerivedFn: Callable type for derived functions.
    Array: Type alias for numpy arrays of float64.
    Number: Type alias for float, list of floats, or numpy arrays.
    Param: Type alias for parameter specifications.
    RetType: Type alias for return types.
    Axes: Type alias for numpy arrays of matplotlib axes.
    ArrayLike: Type alias for numpy arrays or lists of floats.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "Array",
    "ArrayLike",
    "Axes",
    "Derived",
    "DerivedFn",
    "DerivedParameter",
    "DerivedVariable",
    "IntegratorProtocol",
    "McSteadyStates",
    "Number",
    "Param",
    "ProtocolByPars",
    "Reaction",
    "Readout",
    "ResponseCoefficients",
    "ResponseCoefficientsByPars",
    "RetType",
    "SteadyStates",
    "TimeCourseByPars",
    "unwrap",
    "unwrap2",
]

# Re-exporting some types here, because their imports have
# changed between Python versions and I have no interest in
# fixing it in every file
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeVar, cast

import numpy as np
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure
from numpy.typing import NDArray

if TYPE_CHECKING:
    from modelbase2.model import Model

type DerivedFn = Callable[..., float]
type Array = NDArray[np.float64]
type Number = float | list[float] | Array

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

Axes = NDArray[Axis]  # type: ignore
ArrayLike = NDArray[np.float64] | list[float]


def unwrap[T](el: T | None) -> T:
    """Unwraps an optional value, raising an error if the value is None.

    Args:
        el: The value to unwrap. It can be of type T or None.

    Returns:
        The unwrapped value if it is not None.

    Raises:
        ValueError: If the provided value is None.

    """
    if el is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return el


def unwrap2[T1, T2](tpl: tuple[T1 | None, T2 | None]) -> tuple[T1, T2]:
    a, b = tpl
    if a is None or b is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return a, b


class IntegratorProtocol(Protocol):
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
class Reaction:
    fn: DerivedFn
    stoichiometry: Mapping[str, float | Derived]
    args: list[str]

    def get_modifiers(self, model: Model) -> list[str]:
        # FIXME: derived parameters?
        exclude = set(model.parameters) | set(self.stoichiometry)

        return [k for k in self.args if k not in exclude]

    def is_reversible(self, model: Model) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class Readout:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class ResponseCoefficients:
    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class ResponseCoefficientsByPars:
    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class SteadyStates:
    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class McSteadyStates:
    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    mc_parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class TimeCourseByPars:
    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat((self.concs, self.fluxes), axis=1)

    def get_by_name(self, name: str) -> pd.DataFrame:
        return self.results[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        mean = cast(pd.DataFrame, self.results.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        mean = cast(pd.DataFrame, self.results.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)


@dataclass(slots=True)
class ProtocolByPars:
    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    protocol: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat((self.concs, self.fluxes), axis=1)

    def get_by_name(self, name: str) -> pd.DataFrame:
        return self.results[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        mean = cast(pd.DataFrame, self.results.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        mean = cast(pd.DataFrame, self.results.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)
