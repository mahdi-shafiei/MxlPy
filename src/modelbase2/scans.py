"""Parameter Scanning Module.

This module provides functions and classes for performing parameter scans on metabolic models.
It includes functionality for steady-state and time-course simulations, as well as protocol-based simulations.

Classes:
    TimePoint: Represents a single time point in a simulation.
    TimeCourse: Represents a time course in a simulation.

Functions:
    cartesian_product: Generate a cartesian product of the parameter values.
    parameter_scan_ss: Get steady-state results over supplied parameters.
    parameter_scan_time_series: Get time series for each supplied parameter.
    parameter_scan_protocol: Get protocol series for each supplied parameter.
"""

from __future__ import annotations

__all__ = [
    "TimeCourse",
    "TimePoint",
    "cartesian_product",
    "empty_time_course",
    "empty_time_point",
    "parameter_scan_protocol",
    "parameter_scan_ss",
    "parameter_scan_time_series",
]

import itertools as it
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from modelbase2.parallel import Cache, parallelise
from modelbase2.simulator import Simulator
from modelbase2.types import ProtocolByPars, SteadyStates, TimeCourseByPars

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelbase2.model import Model
    from modelbase2.types import Array


def _update_parameters_and[T](
    pars: pd.Series,
    fn: Callable[[Model], T],
    model: Model,
) -> T:
    """Update model parameters and execute a function.

    Args:
        pars: Series containing parameter values to update.
        fn: Function to execute after updating parameters.
        model: Model instance to update.

    Returns:
        Result of the function execution.

    """
    model.update_parameters(pars.to_dict())
    return fn(model)


def _empty_conc_series(model: Model) -> pd.Series:
    """Create an empty concentration series for the model.

    Args:
        model: Model instance to generate the series for.

    Returns:
        pd.Series: Series with NaN values for each model variable.

    """
    return pd.Series(
        data=np.full(shape=len(model.get_variable_names()), fill_value=np.nan),
        index=model.get_variable_names(),
    )


def _empty_flux_series(model: Model) -> pd.Series:
    """Create an empty flux series for the model.

    Args:
        model: Model instance to generate the series for.

    Returns:
        pd.Series: Series with NaN values for each model reaction.

    """
    return pd.Series(
        data=np.full(shape=len(model.get_reaction_names()), fill_value=np.nan),
        index=model.get_reaction_names(),
    )


def _empty_conc_df(model: Model, time_points: Array) -> pd.DataFrame:
    """Create an empty concentration DataFrame for the model over given time points.

    Args:
        model: Model instance to generate the DataFrame for.
        time_points: Array of time points.

    Returns:
        pd.DataFrame: DataFrame with NaN values for each model variable at each time point.

    """
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_variable_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_variable_names(),
    )


def _empty_flux_df(model: Model, time_points: Array) -> pd.DataFrame:
    """Create an empty concentration DataFrame for the model over given time points.

    Args:
        model: Model instance to generate the DataFrame for.
        time_points: Array of time points.

    Returns:
        pd.DataFrame: DataFrame with NaN values for each model variable at each time point.

    """
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_reaction_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_reaction_names(),
    )


def empty_time_point(model: Model) -> tuple[pd.Series, pd.Series]:
    """Create an empty time point for the model.

    Args:
        model: Model instance to generate the time point for.

    """
    return _empty_conc_series(model), _empty_flux_series(model)


def empty_time_course(
    model: Model, time_points: Array
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an empty time course for the model over given time points.

    Args:
        model: Model instance to generate the time course for.
        time_points: Array of time points.

    """
    return _empty_conc_df(model, time_points), _empty_flux_df(model, time_points)


###############################################################################
# Single returns
###############################################################################


@dataclass(slots=True, init=False)
class TimePoint:
    """Represents a single time point in a simulation.

    Attributes:
        concs: Series of concentrations at the time point.
        fluxes: Series of fluxes at the time point.

    Args:
        model: Model instance to generate the time point for.
        concs: DataFrame of concentrations (default: None).
        fluxes: DataFrame of fluxes (default: None).
        idx: Index of the time point in the DataFrame (default: -1).

    """

    concs: pd.Series
    fluxes: pd.Series

    def __init__(
        self,
        model: Model,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
        idx: int = -1,
    ) -> None:
        """Initialize the Scan object.

        Args:
            model (Model): The model object.
            concs (pd.DataFrame | None): DataFrame containing concentration data. If None, an empty concentration series is created.
            fluxes (pd.DataFrame | None): DataFrame containing flux data. If None, an empty flux series is created.
            idx (int, optional): Index to select specific row from concs and fluxes DataFrames. Defaults to -1.

        """
        self.concs = _empty_conc_series(model) if concs is None else concs.iloc[idx]
        self.fluxes = _empty_flux_series(model) if fluxes is None else fluxes.iloc[idx]

    @property
    def results(self) -> pd.Series:
        """Get the combined results of concentrations and fluxes.

        Returns:
            pd.Series: Combined series of concentrations and fluxes.

        """
        return pd.concat((self.concs, self.fluxes), axis=0)


@dataclass(slots=True, init=False)
class TimeCourse:
    """Represents a time course in a simulation.

    Attributes:
        concs: DataFrame of concentrations over time.
        fluxes: DataFrame of fluxes over time.

    Args:
        model: Model instance to generate the time course for.
        time_points: Array of time points.
        concs: DataFrame of concentrations (default: None).
        fluxes: DataFrame of fluxes (default: None).

    """

    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def __init__(
        self,
        model: Model,
        time_points: Array,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
    ) -> None:
        """Initialize the Scan object.

        Args:
            model (Model): The model object.
            time_points (Array): Array of time points.
            concs (pd.DataFrame | None): DataFrame containing concentration data. If None, an empty DataFrame is created.
            fluxes (pd.DataFrame | None): DataFrame containing flux data. If None, an empty DataFrame is created.

        """
        self.concs = _empty_conc_df(model, time_points) if concs is None else concs
        self.fluxes = _empty_flux_df(model, time_points) if fluxes is None else fluxes

    @property
    def results(self) -> pd.DataFrame:
        """Get the combined results of concentrations and fluxes over time.

        Returns:
            pd.DataFrame: Combined DataFrame of concentrations and fluxes.

        """
        return pd.concat((self.concs, self.fluxes), axis=1)


###############################################################################
# Scan returns
###############################################################################


###############################################################################
# Workers
###############################################################################


def _steady_state_worker(
    model: Model,
    y0: dict[str, float] | None,
    *,
    rel_norm: bool,
) -> TimePoint:
    """Simulate the model to steady state and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        rel_norm: Whether to use relative normalization.

    Returns:
        TimePoint: Object containing steady-state concentrations and fluxes.

    """
    c, v = (
        Simulator(model, y0=y0)
        .simulate_to_steady_state(rel_norm=rel_norm)
        .get_full_concs_and_fluxes()
    )
    return TimePoint(model, c, v)


def _time_course_worker(
    model: Model,
    y0: dict[str, float] | None,
    time_points: Array,
) -> TimeCourse:
    """Simulate the model to steady state and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        time_points: Array of time points for the simulation.

    Returns:
        TimePoint: Object containing steady-state concentrations and fluxes.

    """
    c, v = (
        Simulator(model, y0=y0)
        .simulate(time_points=time_points)
        .get_full_concs_and_fluxes()
    )
    return TimeCourse(model, time_points, c, v)


def _protocol_worker(
    model: Model,
    y0: dict[str, float] | None,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
) -> TimeCourse:
    """Simulate the model over a protocol and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        protocol: DataFrame containing the protocol steps.
        time_points_per_step: Number of time points per protocol step (default: 10).

    Returns:
        TimeCourse: Object containing protocol series concentrations and fluxes.

    """
    c, v = (
        Simulator(model, y0=y0)
        .simulate_over_protocol(
            protocol=protocol,
            time_points_per_step=time_points_per_step,
        )
        .get_full_concs_and_fluxes()
    )
    time_points = np.linspace(
        0,
        protocol.index[-1].total_seconds(),
        len(protocol) * time_points_per_step,
    )
    return TimeCourse(model, time_points, c, v)


def cartesian_product(parameters: dict[str, Array]) -> pd.DataFrame:
    """Generate a cartesian product of the parameter values.

    Args:
        parameters: Dictionary containing parameter names and values.

    Returns:
        pd.DataFrame: DataFrame containing the cartesian product of the parameter values.

    """
    return pd.DataFrame(
        it.product(*parameters.values()),
        columns=list(parameters),
    )


def parameter_scan_ss(
    model: Model,
    parameters: pd.DataFrame,
    y0: dict[str, float] | None = None,
    *,
    parallel: bool = True,
    rel_norm: bool = False,
    cache: Cache | None = None,
) -> SteadyStates:
    """Get steady-state results over supplied parameters.

    Args:
        model: Model instance to simulate.
        parameters: DataFrame containing parameter values to scan.
        y0: Initial conditions as a dictionary {variable: value}.
        parallel: Whether to execute in parallel (default: True).
        rel_norm: Whether to use relative normalization (default: False).
        cache: Optional cache to store and retrieve results.

    Returns:
        SteadyStates: Steady-state results for each parameter set.

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                _steady_state_worker,
                y0=y0,
                rel_norm=rel_norm,
            ),
            model=model,
        ),
        inputs=list(parameters.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    concs = pd.DataFrame({k: v.concs.T for k, v in res.items()}).T
    fluxes = pd.DataFrame({k: v.fluxes.T for k, v in res.items()}).T
    idx = (
        pd.Index(parameters.iloc[:, 0])
        if parameters.shape[1] == 1
        else pd.MultiIndex.from_frame(parameters)
    )
    concs.index = idx
    fluxes.index = idx
    return SteadyStates(concs, fluxes, parameters=parameters)


def parameter_scan_time_series(
    model: Model,
    parameters: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    *,
    parallel: bool = True,
    cache: Cache | None = None,
) -> TimeCourseByPars:
    """Get time series for each supplied parameter.

    Args:
        model: Model instance to simulate.
        parameters: DataFrame containing parameter values to scan.
        time_points: Array of time points for the simulation.
        y0: Initial conditions as a dictionary {variable: value}.
        cache: Optional cache to store and retrieve results.
        parallel: Whether to execute in parallel (default: True).

    Returns:
        TimeCourseByPars: Time series results for each parameter set.

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                _time_course_worker,
                time_points=time_points,
                y0=y0,
            ),
            model=model,
        ),
        inputs=list(parameters.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    concs = cast(dict, {k: v.concs for k, v in res.items()})
    fluxes = cast(dict, {k: v.fluxes for k, v in res.items()})
    return TimeCourseByPars(
        parameters=parameters,
        concs=pd.concat(concs, names=["n", "time"]),
        fluxes=pd.concat(fluxes, names=["n", "time"]),
    )


def parameter_scan_protocol(
    model: Model,
    parameters: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
    y0: dict[str, float] | None = None,
    *,
    parallel: bool = True,
    cache: Cache | None = None,
) -> ProtocolByPars:
    """Get protocol series for each supplied parameter.

    Args:
        model: Model instance to simulate.
        parameters: DataFrame containing parameter values to scan.
        protocol: Protocol to follow for the simulation.
        time_points_per_step: Number of time points per protocol step (default: 10).
        y0: Initial conditions as a dictionary {variable: value}.
        cache: Optional cache to store and retrieve results.
        parallel: Whether to execute in parallel (default: True).

    Returns:
        TimeCourseByPars: Protocol series results for each parameter set.

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                _protocol_worker,
                protocol=protocol,
                y0=y0,
                time_points_per_step=time_points_per_step,
            ),
            model=model,
        ),
        inputs=list(parameters.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    concs = cast(dict, {k: v.concs for k, v in res.items()})
    fluxes = cast(dict, {k: v.fluxes for k, v in res.items()})
    return ProtocolByPars(
        parameters=parameters,
        protocol=protocol,
        concs=pd.concat(concs, names=["n", "time"]),
        fluxes=pd.concat(fluxes, names=["n", "time"]),
    )
