"""Parameter Fitting Module for Metabolic Models.

This module provides functions foru fitting model parameters to experimental data,
including both steadyd-state and time-series data fitting capabilities.e

Functions:
    fit_steady_state: Fits parameters to steady-state experimental data
    fit_time_series: Fits parameters to time-series experimental data
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modelbase2.integrators import DefaultIntegrator
from modelbase2.simulator import Simulator
from modelbase2.types import ArrayLike, Callable, IntegratorProtocol, cast

if TYPE_CHECKING:
    from modelbase2.model import Model


def _steady_state_residual(
    par_values: ArrayLike,
    data: pd.Series,
    model: Model,
    y0: dict[str, float],
    par_names: list[str],
    integrator: type[IntegratorProtocol],
) -> float:
    """Calculate residual error between model steady state and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental steady state data
        model: Model instance to simulatep
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use

    Returns:
        float: Root mean square error between model and data

    """
    c_ss, v_ss = (
        Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
            y0=y0,
            integrator=integrator,
        )
        .simulate_to_steady_state()
        .get_full_concs_and_fluxes()
    )
    if c_ss is None or v_ss is None:
        return cast(float, np.inf)
    results_ss = pd.concat((c_ss, v_ss), axis=1)
    diff = data - results_ss.loc[:, data.index]  # type: ignore
    return cast(float, np.sqrt(np.mean(np.square(diff))))


def _time_series_residual(
    par_values: ArrayLike,
    data: pd.DataFrame,
    model: Model,
    y0: dict[str, float],
    par_names: list[str],
    integrator: type[IntegratorProtocol],
) -> float:
    """Calculate residual error between model time course and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental time series data
        model: Model instance to simulate
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use

    Returns:
        float: Root mean square error between model and data

    """
    c_ss, v_ss = (
        Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
            y0=y0,
            integrator=integrator,
        )
        .simulate(time_points=data.index)  # type: ignore
        .get_full_concs_and_fluxes()
    )
    if c_ss is None or v_ss is None:
        return cast(float, np.inf)
    results_ss = pd.concat((c_ss, v_ss), axis=1)
    diff = data - results_ss.loc[:, data.columns]  # type: ignore
    return cast(float, np.sqrt(np.mean(np.square(diff))))


def steady_state(
    model: Model,
    p0: dict[str, float],
    data: pd.Series,
    y0: dict[str, float] | None = None,
    residual_fn: Callable[
        [
            ArrayLike,
            pd.Series,
            Model,
            dict[str, float],
            list[str],
            type[IntegratorProtocol],
        ],
        float,
    ] = _steady_state_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
) -> dict[str, float]:
    """Fit model parameters to steady-state experimental data.

    Args:
        model: Model instance to fit
        data: Experimental steady state data as pandas Series
        p0: Initial parameter guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        residual_fn: Function to calculate fitting error (default: _steady_state_residual)
        integrator: ODE integrator class (default: DefaultIntegrator)

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-12, 1e6] for all parameters

    """
    par_names = list(p0.keys())
    x0 = list(p0.values())

    # Copy to restore
    p_orig = model.parameters

    res = dict(
        zip(
            par_names,
            minimize(
                residual_fn,
                x0=x0,
                args=(data, model, y0, par_names, integrator),
                bounds=[(1e-12, 1e6) for _ in range(len(p0))],
                method="L-BFGS-B",
            ).x,
            strict=True,
        )
    )

    # Restore
    model.update_parameters(p_orig)
    return res


def time_series(
    model: Model,
    p0: dict[str, float],
    data: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: Callable[
        [
            ArrayLike,
            pd.DataFrame,
            Model,
            dict[str, float],
            list[str],
            type[IntegratorProtocol],
        ],
        float,
    ] = _time_series_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
) -> dict[str, float]:
    """Fit model parameters to time-series experimental data.

    Args:
        model: Model instance to fit
        data: Experimental time series data as pandas DataFrame
        p0: Initial parameter guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        residual_fn: Function to calculate fitting error (default: _time_series_residual)
        integrator: ODE integrator class (default: DefaultIntegrator)

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-12, 1e6] for all parameters

    """
    par_names = list(p0.keys())
    x0 = list(p0.values())
    p_orig = model.parameters

    res = dict(
        zip(
            par_names,
            minimize(
                residual_fn,
                x0=x0,
                args=(data, model, y0, par_names, integrator),
                bounds=[(1e-12, 1e6) for _ in range(len(p0))],
                method="L-BFGS-B",
            ).x,
            strict=True,
        )
    )
    model.update_parameters(p_orig)
    return res
