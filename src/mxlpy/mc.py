"""Monte Carlo Analysis (MC) Module for Metabolic Models.

This module provides functions for performing Monte Carlo analysis on metabolic models.
It includes functionality for steady-state and time-course simulations, as well as
response coefficient calculations.

Functions:
    steady_state: Perform Monte Carlo analysis for steady-state simulations
    time_course: Perform Monte Carlo analysis for time-course simulations
    time_course_over_protocol: Perform Monte Carlo analysis for time-course simulations over a protocol
    parameter_scan_ss: Perform Monte Carlo analysis for steady-state parameter scans
    compound_elasticities: Calculate compound elasticities using Monte Carlo analysis
    parameter_elasticities: Calculate parameter elasticities using Monte Carlo analysis
    response_coefficients: Calculate response coefficients using Monte Carlo analysis
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd

from mxlpy import mca, scan
from mxlpy.integrators import DefaultIntegrator
from mxlpy.parallel import Cache, parallelise
from mxlpy.scan import (
    ProtocolWorker,
    SteadyStateWorker,
    TimeCourseWorker,
    _protocol_worker,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and,
)
from mxlpy.types import (
    ArrayLike,
    IntegratorProtocol,
    McSteadyStates,
    ProtocolByPars,
    ResponseCoefficientsByPars,
    SteadyStates,
    TimeCourseByPars,
)

__all__ = [
    "ParameterScanWorker",
    "parameter_elasticities",
    "response_coefficients",
    "scan_steady_state",
    "steady_state",
    "time_course",
    "time_course_over_protocol",
    "variable_elasticities",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model
    from mxlpy.types import Array

__ALL__ = [
    "steady_state",
    "time_course",
    "time_course_over_protocol",
    "parameter_scan_ss",
    "compound_elasticities",
    "parameter_elasticities",
    "response_coefficients",
]


class ParameterScanWorker(Protocol):
    """Protocol for the parameter scan worker function."""

    def __call__(
        self,
        model: Model,
        y0: dict[str, float] | None,
        *,
        parameters: pd.DataFrame,
        rel_norm: bool,
        integrator: Callable[[Callable, ArrayLike], IntegratorProtocol],
    ) -> SteadyStates:
        """Call the worker function."""
        ...


def _parameter_scan_worker(
    model: Model,
    y0: dict[str, float] | None,
    *,
    parameters: pd.DataFrame,
    rel_norm: bool,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol],
) -> SteadyStates:
    """Worker function for parallel steady state scanning across parameter sets.

    This function executes a parameter scan for steady state solutions for a
    given model and parameter combinations. It's designed to be used as a worker
    in parallel processing.

    Args: model : Model
        The model object to analyze
    y0 : dict[str, float] | None
        Initial conditions for the solver. If None, default initial conditions
        are used.
    parameters : pd.DataFrame
        DataFrame containing parameter combinations to scan over. Each row
        represents one parameter set.
    rel_norm : bool
        Whether to use relative normalization in the steady state calculations

    Returns:
        SteadyStates
            Object containing the steady state solutions for the given parameter
            combinations

    """
    return scan.steady_state(
        model,
        parameters=parameters,
        y0=y0,
        parallel=False,
        rel_norm=rel_norm,
        integrator=integrator,
    )


def steady_state(
    model: Model,
    mc_parameters: pd.DataFrame,
    *,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = False,
    worker: SteadyStateWorker = _steady_state_worker,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol] = DefaultIntegrator,
) -> SteadyStates:
    """Monte-carlo scan of steady states.

    Examples:
        >>> steady_state(model, mc_parameters)
        p    t     x      y
        0    0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1    0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03

    Returns:
        SteadyStates: Object containing the steady state solutions for the given parameter

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
                y0=y0,
                rel_norm=rel_norm,
                integrator=integrator,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    return SteadyStates(
        variables=pd.concat({k: v.variables for k, v in res.items()}, axis=1).T,
        fluxes=pd.concat({k: v.fluxes for k, v in res.items()}, axis=1).T,
        parameters=mc_parameters,
    )


def time_course(
    model: Model,
    time_points: Array,
    mc_parameters: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    worker: TimeCourseWorker = _time_course_worker,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol] = DefaultIntegrator,
) -> TimeCourseByPars:
    """MC time course.

    Examples:
        >>> time_course(model, time_points, mc_parameters)
        p    t     x      y
        0   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
    Returns:
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_parameters, #variables)

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
                time_points=time_points,
                y0=y0,
                integrator=integrator,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )

    return TimeCourseByPars(
        parameters=mc_parameters,
        variables=pd.concat({k: v.variables.T for k, v in res.items()}, axis=1).T,
        fluxes=pd.concat({k: v.fluxes.T for k, v in res.items()}, axis=1).T,
    )


def time_course_over_protocol(
    model: Model,
    protocol: pd.DataFrame,
    mc_parameters: pd.DataFrame,
    y0: dict[str, float] | None = None,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol] = DefaultIntegrator,
) -> ProtocolByPars:
    """MC time course.

    Examples:
        >>> time_course_over_protocol(model, protocol, mc_parameters)
        p    t     x      y
        0   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03

    Returns:
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_parameters, #variables)

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
                protocol=protocol,
                y0=y0,
                integrator=integrator,
                time_points_per_step=time_points_per_step,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v.variables.T for k, v in res.items()}
    fluxes = {k: v.fluxes.T for k, v in res.items()}
    return ProtocolByPars(
        variables=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
        parameters=mc_parameters,
        protocol=protocol,
    )


def scan_steady_state(
    model: Model,
    parameters: pd.DataFrame,
    mc_parameters: pd.DataFrame,
    *,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = False,
    worker: ParameterScanWorker = _parameter_scan_worker,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol] = DefaultIntegrator,
) -> McSteadyStates:
    """Parameter scan of mc distributed steady states.

    Examples:
        >>> scan_steady_state(
        ...     model,
        ...     parameters=pd.DataFrame({"k1": np.linspace(0, 1, 3)}),
        ...     mc_parameters=mc_parameters,
        ... ).variables
                  x     y
          k1
        0 0.0 -0.00 -0.00
          0.5  0.44  0.20
          1.0  0.88  0.40
        1 0.0 -0.00 -0.00
          0.5  0.45  0.14
          1.0  0.90  0.28



    Args:
        model: The model to analyze
        parameters: DataFrame containing parameter combinations to scan over
        mc_parameters: DataFrame containing Monte Carlo parameter sets
        y0: Initial conditions for the solver
        max_workers: Maximum number of workers for parallel processing
        cache: Cache object for storing results
        rel_norm: Whether to use relative normalization in the steady state calculations
        worker: Worker function for parallel steady state scanning across parameter sets
        integrator: Function producing an integrator for the simulation.

    Returns:
        McSteadyStates: Object containing the steady state solutions for the given parameter

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                worker,
                parameters=parameters,
                y0=y0,
                rel_norm=rel_norm,
                integrator=integrator,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    concs = {k: v.variables.T for k, v in res.items()}
    fluxes = {k: v.fluxes.T for k, v in res.items()}
    return McSteadyStates(
        variables=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
        parameters=parameters,
        mc_parameters=mc_parameters,
    )


def variable_elasticities(
    model: Model,
    variables: list[str],
    concs: dict[str, float],
    mc_parameters: pd.DataFrame,
    *,
    time: float = 0,
    cache: Cache | None = None,
    max_workers: int | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate variable elasticities using Monte Carlo analysis.

    Examples:
        >>> variable_elasticities(
        ...     model,
        ...     variables=["x1", "x2"],
        ...     concs={"x1": 1, "x2": 2},
        ...     mc_parameters=mc_parameters
        ... )
                 x1     x2
        0   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4
        1   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4

    Args:
        model: The model to analyze
        variables: List of variables for which to calculate elasticities
        concs: Dictionary of concentrations for the model
        mc_parameters: DataFrame containing Monte Carlo parameter sets
        time: Time point for the analysis
        cache: Cache object for storing results
        max_workers: Maximum number of workers for parallel processing
        normalized: Whether to use normalized elasticities
        displacement: Displacement for finite difference calculations

    Returns:
        pd.DataFrame: DataFrame containing the compound elasticities for the given variables

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                mca.variable_elasticities,
                variables=variables,
                concs=concs,
                time=time,
                displacement=displacement,
                normalized=normalized,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    return cast(pd.DataFrame, pd.concat(res))


def parameter_elasticities(
    model: Model,
    parameters: list[str],
    concs: dict[str, float],
    mc_parameters: pd.DataFrame,
    *,
    time: float = 0,
    cache: Cache | None = None,
    max_workers: int | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate parameter elasticities using Monte Carlo analysis.

    Examples:
        >>> parameter_elasticities(
        ...     model,
        ...     variables=["p1", "p2"],
        ...     concs={"x1": 1, "x2": 2},
        ...     mc_parameters=mc_parameters
        ... )
                 p1     p2
        0   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4
        1   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4

    Args:
        model: The model to analyze
        parameters: List of parameters for which to calculate elasticities
        concs: Dictionary of concentrations for the model
        mc_parameters: DataFrame containing Monte Carlo parameter sets
        time: Time point for the analysis
        cache: Cache object for storing results
        max_workers: Maximum number of workers for parallel processing
        normalized: Whether to use normalized elasticities
        displacement: Displacement for finite difference calculations

    Returns:
        pd.DataFrame: DataFrame containing the parameter elasticities for the given variables

    """
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                mca.parameter_elasticities,
                parameters=parameters,
                concs=concs,
                time=time,
                displacement=displacement,
                normalized=normalized,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    return cast(pd.DataFrame, pd.concat(res))


def response_coefficients(
    model: Model,
    parameters: list[str],
    mc_parameters: pd.DataFrame,
    *,
    y0: dict[str, float] | None = None,
    cache: Cache | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    rel_norm: bool = False,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol] = DefaultIntegrator,
) -> ResponseCoefficientsByPars:
    """Calculate response coefficients using Monte Carlo analysis.

    Examples:
        >>> response_coefficients(
        ...     model,
        ...     parameters=["vmax1", "vmax2"],
        ...     mc_parameters=mc_parameters,
        ... ).variables
                    x1    x2
        0 vmax_1  0.01  0.01
          vmax_2  0.02  0.02
        1 vmax_1  0.03  0.03
          vmax_2  0.04  0.04

    Args:
        model: The model to analyze
        parameters: List of parameters for which to calculate elasticities
        mc_parameters: DataFrame containing Monte Carlo parameter sets
        y0: Initial conditions for the solver
        cache: Cache object for storing results
        normalized: Whether to use normalized elasticities
        displacement: Displacement for finite difference calculations
        disable_tqdm: Whether to disable the tqdm progress bar
        max_workers: Maximum number of workers for parallel processing
        rel_norm: Whether to use relative normalization in the steady state calculations
        integrator: Function producing an integrator for the simulation.

    Returns:
        ResponseCoefficientsByPars: Object containing the response coefficients for the given parameters

    """
    res = parallelise(
        fn=partial(
            _update_parameters_and,
            fn=partial(
                mca.response_coefficients,
                parameters=parameters,
                y0=y0,
                normalized=normalized,
                displacement=displacement,
                rel_norm=rel_norm,
                disable_tqdm=disable_tqdm,
                parallel=False,
                integrator=integrator,
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )

    return ResponseCoefficientsByPars(
        variables=cast(
            pd.DataFrame, pd.concat({k: v.variables for k, v in res.items()})
        ),
        fluxes=cast(pd.DataFrame, pd.concat({k: v.fluxes for k, v in res.items()})),
        parameters=mc_parameters,
    )
