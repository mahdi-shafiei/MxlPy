"""Fit model to data

- steady-state concentrations / fluxes
- time series concentrations / fluxes
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
