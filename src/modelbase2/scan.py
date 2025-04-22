"""Parameter Scanning Module.

This module provides functions and classes for performing parameter scans on metabolic models.
It includes functionality for steady-state and time-course simulations, as well as protocol-based simulations.

Classes:
    TimePoint: Represents a single time point in a simulation.
    TimeCourse: Represents a time course in a simulation.

Functions:
    parameter_scan_ss: Get steady-state results over supplied parameters.
    parameter_scan_time_course: Get time course for each supplied parameter.
    parameter_scan_protocol: Get protocol course for each supplied parameter.
"""

from __future__ import annotations

__all__ = [
    "ProtocolWorker",
    "SteadyStateWorker",
    "TimeCourse",
    "TimeCourseWorker",
    "TimePoint",
    "steady_state",
    "time_course",
    "time_course_over_protocol",
]

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, Self, cast

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


###############################################################################
# Single returns
###############################################################################


@dataclass(slots=True)
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

    @classmethod
    def from_scan(
        cls,
        model: Model,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
        idx: int = -1,
    ) -> Self:
        """Initialize the Scan object.

        Args:
            model (Model): The model object.
            concs (pd.DataFrame | None): DataFrame containing concentration data. If None, an empty concentration series is created.
            fluxes (pd.DataFrame | None): DataFrame containing flux data. If None, an empty flux series is created.
            idx (int, optional): Index to select specific row from concs and fluxes DataFrames. Defaults to -1.

        """
        return cls(
            concs=_empty_conc_series(model) if concs is None else concs.iloc[idx],
            fluxes=_empty_flux_series(model) if fluxes is None else fluxes.iloc[idx],
        )

    @property
    def results(self) -> pd.Series:
        """Get the combined results of concentrations and fluxes.

        Example:
            >>> time_point.results
            x1    1.0
            x2    0.5
            v1    0.1
            v2    0.2

        Returns:
            pd.Series: Combined series of concentrations and fluxes.

        """
        return pd.concat((self.concs, self.fluxes), axis=0)


@dataclass(slots=True)
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

    @classmethod
    def from_scan(
        cls,
        model: Model,
        time_points: Array,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
    ) -> Self:
        """Initialize the Scan object.

        Args:
            model (Model): The model object.
            time_points (Array): Array of time points.
            concs (pd.DataFrame | None): DataFrame containing concentration data. If None, an empty DataFrame is created.
            fluxes (pd.DataFrame | None): DataFrame containing flux data. If None, an empty DataFrame is created.

        """
        return cls(
            _empty_conc_df(model, time_points) if concs is None else concs,
            _empty_flux_df(model, time_points) if fluxes is None else fluxes,
        )

    @property
    def results(self) -> pd.DataFrame:
        """Get the combined results of concentrations and fluxes over time.

        Examples:
            >>> time_course.results
            Time   x1     x2     v1     v2
            0.0   1.0   1.00   1.00   1.00
            0.1   0.9   0.99   0.99   0.99
            0.2   0.8   0.99   0.99   0.99

        Returns:
            pd.DataFrame: Combined DataFrame of concentrations and fluxes.

        """
        return pd.concat((self.concs, self.fluxes), axis=1)


###############################################################################
# Workers
###############################################################################


class SteadyStateWorker(Protocol):
    """Worker function for steady-state simulations."""

    def __call__(
        self,
        model: Model,
        y0: dict[str, float] | None,
        *,
        rel_norm: bool,
    ) -> TimePoint:
        """Call the worker function."""
        ...


class TimeCourseWorker(Protocol):
    """Worker function for time-course simulations."""

    def __call__(
        self,
        model: Model,
        y0: dict[str, float] | None,
        time_points: Array,
    ) -> TimeCourse:
        """Call the worker function."""
        ...


class ProtocolWorker(Protocol):
    """Worker function for protocol-based simulations."""

    def __call__(
        self,
        model: Model,
        y0: dict[str, float] | None,
        protocol: pd.DataFrame,
        time_points_per_step: int = 10,
    ) -> TimeCourse:
        """Call the worker function."""
        ...


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
    try:
        c, v = (
            Simulator(model, y0=y0)
            .simulate_to_steady_state(rel_norm=rel_norm)
            .get_full_concs_and_fluxes()
        )
    except ZeroDivisionError:
        c = None
        v = None
    return TimePoint.from_scan(model, c, v)


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
    try:
        c, v = (
            Simulator(model, y0=y0)
            .simulate_time_course(time_points=time_points)
            .get_full_concs_and_fluxes()
        )
    except ZeroDivisionError:
        c = None
        v = None
    return TimeCourse.from_scan(model, time_points, c, v)


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
    return TimeCourse.from_scan(model, time_points, c, v)


def steady_state(
    model: Model,
    parameters: pd.DataFrame,
    y0: dict[str, float] | None = None,
    *,
    parallel: bool = True,
    rel_norm: bool = False,
    cache: Cache | None = None,
    worker: SteadyStateWorker = _steady_state_worker,
) -> SteadyStates:
    """Get steady-state results over supplied parameters.

    Args:
        model: Model instance to simulate.
        parameters: DataFrame containing parameter values to scan.
        y0: Initial conditions as a dictionary {variable: value}.
        parallel: Whether to execute in parallel (default: True).
        rel_norm: Whether to use relative normalization (default: False).
        cache: Optional cache to store and retrieve results.
        worker: Worker function to use for the simulation.

    Returns:
        SteadyStates: Steady-state results for each parameter set.

    Examples:
        >>> steady_state(
        >>>     model,
        >>>     parameters=pd.DataFrame({"k1": np.linspace(1, 2, 3)})
        >>> ).concs
        idx      x      y
        1.0   0.50   1.00
        1.5   0.75   1.50
        2.0   1.00   2.00

        >>> steady_state(
        >>>     model,
        >>>     parameters=cartesian_product({"k1": [1, 2], "k2": [3, 4]})
        >>> ).concs

        | idx    |    x |   y |
        | (1, 3) | 0.33 |   1 |
        | (1, 4) | 0.25 |   1 |
        | (2, 3) | 0.66 |   2 |
        | (2, 4) | 0.5  |   2 |

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
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
    return SteadyStates(concs=concs, fluxes=fluxes, parameters=parameters)


def time_course(
    model: Model,
    parameters: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    *,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: TimeCourseWorker = _time_course_worker,
) -> TimeCourseByPars:
    """Get time course for each supplied parameter.

    Examples:
        >>> time_course(
        >>>     model,
        >>>     parameters=pd.DataFrame({"k1": [1, 1.5, 2]}),
        >>>     time_points=np.linspace(0, 1, 3)
        >>> ).concs

        | (n, time) |        x |       y |
        |:----------|---------:|--------:|
        | (0, 0.0)  | 1        | 1       |
        | (0, 0.5)  | 0.68394  | 1.23865 |
        | (0, 1.0)  | 0.567668 | 1.23254 |
        | (1, 0.0)  | 1        | 1       |
        | (1, 0.5)  | 0.84197  | 1.31606 |
        | (1, 1.0)  | 0.783834 | 1.43233 |
        | (2, 0.0)  | 1        | 1       |
        | (2, 0.5)  | 1        | 1.39347 |
        | (2, 1.0)  | 1        | 1.63212 |

        >>> time_course(
        >>>     model,
        >>>     parameters=cartesian_product({"k1": [1, 2], "k2": [3, 4]}),
        >>>     time_points=[0.0, 0.5, 1.0],
        >>> ).concs

        | (n, time) |        x |      y |
        |:----------|---------:|-------:|
        | (0, 0.0)  | 1        | 1      |
        | (0, 0.5)  | 0.482087 | 1.3834 |
        | (1, 0.0)  | 1        | 1      |
        | (1, 0.5)  | 0.351501 | 1.4712 |
        | (2, 0.0)  | 1        | 1      |

    Args:
        model: Model instance to simulate.
        parameters: DataFrame containing parameter values to scan.
        time_points: Array of time points for the simulation.
        y0: Initial conditions as a dictionary {variable: value}.
        cache: Optional cache to store and retrieve results.
        parallel: Whether to execute in parallel (default: True).
        worker: Worker function to use for the simulation.

    Returns:
        TimeCourseByPars: Time series results for each parameter set.


    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
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


def time_course_over_protocol(
    model: Model,
    parameters: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
    y0: dict[str, float] | None = None,
    *,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
) -> ProtocolByPars:
    """Get protocol series for each supplied parameter.

    Examples:
        >>> scan.time_course_over_protocol(
        ...     model,
        ...     parameters=pd.DataFrame({"k2": np.linspace(1, 2, 11)}),
        ...     protocol=make_protocol(
        ...         {
        ...             1: {"k1": 1},
        ...             2: {"k1": 2},
        ...         }
        ...     ),
        ... )

    Args:
        model: Model instance to simulate.
        parameters: DataFrame containing parameter values to scan.
        protocol: Protocol to follow for the simulation.
        time_points_per_step: Number of time points per protocol step (default: 10).
        y0: Initial conditions as a dictionary {variable: value}.
        parallel: Whether to execute in parallel (default: True).
        cache: Optional cache to store and retrieve results.
        worker: Worker function to use for the simulation.

    Returns:
        TimeCourseByPars: Protocol series results for each parameter set.

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
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
