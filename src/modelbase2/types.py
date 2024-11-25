from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Any, ParamSpec, Protocol, Self, TypeVar, cast

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


def unwrap(el: T | None) -> T:
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


def default_if_none(el: T | None, default: T) -> T:
    return default if el is None else el


class ModelProtocol(Protocol):
    # Parameters
    def get_parameter_names(self) -> list[str]: ...
    @property
    def parameters(self) -> dict[str, float]: ...
    def update_parameters(self, parameters: dict[str, float]) -> Self: ...
    @property
    def derived_parameters(self) -> dict[str, DerivedParameter]: ...

    # Variables
    def get_variable_names(self) -> list[str]: ...
    def get_derived_variable_names(self) -> list[str]: ...
    def get_readout_names(self) -> list[str]: ...
    @property
    def derived_variables(self) -> dict[str, DerivedVariable]: ...

    # Reactions
    def get_reaction_names(self) -> list[str]: ...
    @property
    def reactions(self) -> dict[str, Reaction]: ...

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
    def get_initial_conditions(self) -> dict[str, float]: ...

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
class Reaction:
    fn: DerivedFn
    stoichiometry: Mapping[str, float | Derived]
    args: list[str]

    def get_modifiers(self, model: ModelProtocol) -> list[str]:
        # FIXME: derived parameters?
        exclude = set(model.parameters) | set(self.stoichiometry)

        return [k for k in self.args if k not in exclude]

    def is_reversible(self, model: ModelProtocol) -> bool:
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
