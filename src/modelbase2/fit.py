"""Fit model to data

- steady-state concentrations / fluxes
- time series concentrations / fluxes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from modelbase2.integrators import DefaultIntegrator
from modelbase2.simulator import Simulator
from modelbase2.types import ArrayLike, Callable, IntegratorProtocol, cast

if TYPE_CHECKING:
    from modelbase2.types import ModelProtocol


def _steady_state_concs_residual(
    par_values: ArrayLike,
    data: ArrayLike,
    model: ModelProtocol,
    y0: dict[str, float],
    par_names: list[str],
    integrator: type[IntegratorProtocol],
) -> float:
    if (
        y_ss := Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
            y0=y0,
            integrator=integrator,
        )
        .simulate_to_steady_state()
        .get_concs()
    ) is None:
        return cast(float, np.inf)
    return cast(float, np.sqrt(np.mean(np.square(data - y_ss.to_numpy()))))


def _steady_state_fluxes_residual(
    par_values: ArrayLike,
    data: ArrayLike,
    model: ModelProtocol,
    y0: dict[str, float],
    par_names: list[str],
    integrator: type[IntegratorProtocol],
) -> float:
    if (
        y_ss := Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
            y0=y0,
            integrator=integrator,
        )
        .simulate_to_steady_state()
        .get_fluxes()
    ) is None:
        return cast(float, np.inf)
    return cast(float, np.sqrt(np.mean(np.square(data - y_ss.to_numpy()))))


def _time_series_concs_residual(
    par_values: ArrayLike,
    data: ArrayLike,
    time_points: ArrayLike,
    model: ModelProtocol,
    y0: dict[str, float],
    par_names: list[str],
    integrator: type[IntegratorProtocol],
) -> float:
    if (
        y := Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
            y0=y0,
            integrator=integrator,
        )
        .simulate(time_points=time_points)
        .get_concs()
    ) is None:
        return cast(float, np.inf)
    return cast(float, np.sqrt(np.mean(np.square(data - y.to_numpy()))))


def steady_state_variables(
    model: ModelProtocol,
    p0: dict[str, float],
    y0: dict[str, float],
    data: ArrayLike,
    residual_fn: Callable[
        [
            ArrayLike,
            ArrayLike,
            ModelProtocol,
            dict[str, float],
            list[str],
            type[IntegratorProtocol],
        ],
        float,
    ] = _steady_state_concs_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
) -> dict[str, float]:
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


def time_series_variables(
    model: ModelProtocol,
    p0: dict[str, float],
    y0: dict[str, float],
    data: ArrayLike,
    time_points: ArrayLike,
    residual_fn: Callable[
        [
            ArrayLike,
            ArrayLike,
            ArrayLike,
            ModelProtocol,
            dict[str, float],
            list[str],
            type[IntegratorProtocol],
        ],
        float,
    ] = _time_series_concs_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
) -> dict[str, float]:
    par_names = list(p0.keys())
    x0 = list(p0.values())
    p_orig = model.parameters
    if len(data) != len(time_points):
        raise ValueError

    res = dict(
        zip(
            par_names,
            minimize(
                residual_fn,
                x0=x0,
                args=(data, time_points, model, y0, par_names, integrator),
                bounds=[(1e-12, 1e6) for _ in range(len(p0))],
                method="L-BFGS-B",
            ).x,
            strict=True,
        )
    )
    model.update_parameters(p_orig)
    return res
