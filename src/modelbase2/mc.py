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

from modelbase2 import mca, scan
from modelbase2.parallel import Cache, parallelise
from modelbase2.scan import (
    ProtocolWorker,
    SteadyStateWorker,
    TimeCourseWorker,
    _protocol_worker,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and,
)
from modelbase2.types import (
    McSteadyStates,
    ProtocolByPars,
    ResponseCoefficientsByPars,
    SteadyStates,
    TimeCourseByPars,
)

__all__ = ["ParameterScanWorker", "compound_elasticities", "parameter_elasticities", "response_coefficients", "scan_steady_state", "steady_state", "time_course", "time_course_over_protocol"]

if TYPE_CHECKING:
    from modelbase2.model import Model
    from modelbase2.types import Array

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
    ) -> SteadyStates:
        """Call the worker function."""
        ...


def _parameter_scan_worker(
    model: Model,
    y0: dict[str, float] | None,
    *,
    parameters: pd.DataFrame,
    rel_norm: bool,
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
    )


def steady_state(
    model: Model,
    mc_parameters: pd.DataFrame,
    *,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = True,
    worker: SteadyStateWorker = _steady_state_worker,
) -> SteadyStates:
    """Monte-carlo scan of steady states.

    Returns:
        SteadyStates: Object containing the steady state solutions for the given parameter

    Example:
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
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v.concs for k, v in res.items()}
    fluxes = {k: v.fluxes for k, v in res.items()}
    return SteadyStates(
        concs=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
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
) -> TimeCourseByPars:
    """MC time course.

    Returns
    -------
    tuple[concentrations, fluxes] using pandas multiindex
    Both dataframes are of shape (#time_points * #mc_parameters, #variables)

    E.g.
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
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v.concs.T for k, v in res.items()}
    fluxes = {k: v.fluxes.T for k, v in res.items()}
    return TimeCourseByPars(
        parameters=mc_parameters,
        concs=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
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
) -> ProtocolByPars:
    """MC time course.

    Returns
    -------
    tuple[concentrations, fluxes] using pandas multiindex
    Both dataframes are of shape (#time_points * #mc_parameters, #variables)

    E.g.
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
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v.concs.T for k, v in res.items()}
    fluxes = {k: v.fluxes.T for k, v in res.items()}
    return ProtocolByPars(
        concs=pd.concat(concs, axis=1).T,
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
) -> McSteadyStates:
    """Parameter scan of mc distributed steady states.

    Args:
        model: The model to analyze
        parameters: DataFrame containing parameter combinations to scan over
        mc_parameters: DataFrame containing Monte Carlo parameter sets
        y0: Initial conditions for the solver
        max_workers: Maximum number of workers for parallel processing
        cache: Cache object for storing results
        rel_norm: Whether to use relative normalization in the steady state calculations
        worker: Worker function for parallel steady state scanning across parameter sets

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
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    concs = {k: v.concs.T for k, v in res.items()}
    fluxes = {k: v.fluxes.T for k, v in res.items()}
    return McSteadyStates(
        concs=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
        parameters=parameters,
        mc_parameters=mc_parameters,
    )


def compound_elasticities(
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
    """Calculate compound elasticities using Monte Carlo analysis.

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
) -> ResponseCoefficientsByPars:
    """Calculate response coefficients using Monte Carlo analysis.

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
            ),
            model=model,
        ),
        inputs=list(mc_parameters.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )

    crcs = {k: v.concs for k, v in res.items()}
    frcs = {k: v.fluxes for k, v in res.items()}

    return ResponseCoefficientsByPars(
        concs=cast(pd.DataFrame, pd.concat(crcs)),
        fluxes=cast(pd.DataFrame, pd.concat(frcs)),
        parameters=mc_parameters,
    )
