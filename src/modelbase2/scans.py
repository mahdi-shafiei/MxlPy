"""Scans

| parameter_scan_ss          | Get steady-state results over supplied parameters |
| parameter_scan_time_series | Get time series for each supplied parameter       |
| parameter_scan_protocol    | Get protocol series for each supplied parameter   |
"""

from __future__ import annotations

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
    from modelbase2.types import Array, T


def _update_parameters_and(
    pars: pd.Series,
    fn: Callable[[Model], T],
    model: Model,
) -> T:
    model.update_parameters(pars.to_dict())
    return fn(model)


def _empty_conc_series(model: Model) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.get_variable_names()), fill_value=np.nan),
        index=model.get_variable_names(),
    )


def _empty_flux_series(model: Model) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.get_reaction_names()), fill_value=np.nan),
        index=model.get_reaction_names(),
    )


def _empty_conc_df(model: Model, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_variable_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_variable_names(),
    )


def _empty_flux_df(model: Model, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_reaction_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_reaction_names(),
    )


def empty_time_point(model: Model) -> tuple[pd.Series, pd.Series]:
    return _empty_conc_series(model), _empty_flux_series(model)


def empty_time_course(
    model: Model, time_points: Array
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _empty_conc_df(model, time_points), _empty_flux_df(model, time_points)


###############################################################################
# Single returns
###############################################################################


@dataclass(slots=True, init=False)
class TimePoint:
    concs: pd.Series
    fluxes: pd.Series

    def __init__(
        self,
        model: Model,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
        idx: int = -1,
    ) -> None:
        self.concs = _empty_conc_series(model) if concs is None else concs.iloc[idx]
        self.fluxes = _empty_flux_series(model) if fluxes is None else fluxes.iloc[idx]

    @property
    def results(self) -> pd.Series:
        return pd.concat((self.concs, self.fluxes), axis=0)


@dataclass(slots=True, init=False)
class TimeCourse:
    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def __init__(
        self,
        model: Model,
        time_points: Array,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
    ) -> None:
        self.concs = _empty_conc_df(model, time_points) if concs is None else concs
        self.fluxes = _empty_flux_df(model, time_points) if fluxes is None else fluxes

    @property
    def results(self) -> pd.DataFrame:
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
    if parameters.shape[1] == 1:
        idx = pd.Index(parameters.iloc[:, 0])
    else:
        idx = pd.MultiIndex.from_frame(parameters)
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
