from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import pandas as pd

from modelbase2 import mca, scans
from modelbase2.parallel import Cache, parallelise
from modelbase2.scans import (
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

if TYPE_CHECKING:
    from modelbase2.types import Array, ModelProtocol


def _parameter_scan_worker(
    model: ModelProtocol,
    y0: dict[str, float] | None,
    *,
    parameters: pd.DataFrame,
    rel_norm: bool,
) -> SteadyStates:
    return scans.parameter_scan_ss(
        model,
        parameters=parameters,
        y0=y0,
        parallel=False,
        rel_norm=rel_norm,
    )


def steady_state(
    model: ModelProtocol,
    mc_parameters: pd.DataFrame,
    *,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = True,
) -> SteadyStates:
    """MC time course

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
                _steady_state_worker,
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
    model: ModelProtocol,
    time_points: Array,
    mc_parameters: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> TimeCourseByPars:
    """MC time course

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
                _time_course_worker,
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
    model: ModelProtocol,
    protocol: pd.DataFrame,
    mc_parameters: pd.DataFrame,
    y0: dict[str, float] | None = None,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> ProtocolByPars:
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


def parameter_scan_ss(
    model: ModelProtocol,
    parameters: pd.DataFrame,
    mc_parameters: pd.DataFrame,
    *,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = False,
) -> McSteadyStates:
    res = parallelise(
        partial(
            _update_parameters_and,
            fn=partial(
                _parameter_scan_worker,
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
    model: ModelProtocol,
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
    model: ModelProtocol,
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
    model: ModelProtocol,
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
