from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from modelbase2.integrators import DefaultIntegrator, IntegratorProtocol
from modelbase2.simulator import Simulator
from modelbase2.types import ArrayLike, Callable, cast

if TYPE_CHECKING:
    from modelbase2.models import ModelProtocol


def _steady_state_residual(
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
        ).simulate_to_steady_state()
    ) is None:
        return cast(float, np.inf)
    return cast(float, np.sqrt(np.mean(np.square(data - y_ss.to_numpy()))))


def steady_state(
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
    ] = _steady_state_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
) -> dict[str, float]:
    par_names = list(p0.keys())
    x0 = list(p0.values())

    # Copy to restore
    p_orig = model.get_parameters().copy()

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


def _time_series_residual(
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
        ).simulate(time_points=time_points)
    ) is None:
        return cast(float, np.inf)
    return cast(float, np.sqrt(np.mean(np.square(data - y.to_numpy()))))


def time_series(
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
    ] = _time_series_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
) -> dict[str, float]:
    par_names = list(p0.keys())
    x0 = list(p0.values())
    p_orig = model.get_parameters().copy()
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
