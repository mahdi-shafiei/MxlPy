from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns

from modelbase2.mc._parallel import Cache, parallelise
from modelbase2.mc._plot import _fig_ax_if_neccessary, _plot_line_mean_std
from modelbase2.ode import Model, Simulator
from modelbase2.types import Array, Axis, default_if_none
from modelbase2.utils.plotting import grid


@dataclass
class McTimeCourse:
    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def plot_line_mean_std(
        self,
        df: Literal["concs", "fluxes"],
        var: str,
        label: str | None = None,
        color: str | None = None,
        ax: Axis | None = None,
    ) -> Axis:
        _, ax = _fig_ax_if_neccessary(ax=ax)

        _plot_line_mean_std(
            ax=ax,
            df=(self.concs if df == "concs" else self.fluxes)[var].unstack().T,
            color=color,
            label=var if label is None else label,
        )
        grid(ax=ax)
        return ax


@dataclass
class McTimeProtocol(McTimeCourse):
    protocol: pd.DataFrame

    def shade_protocol(
        self,
        key: str,
        ax: Axis,
        *,
        cmap_name: str = "Greys_r",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 0.5,
        add_legend: bool = True,
    ) -> None:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        from matplotlib.legend import Legend
        from matplotlib.patches import Patch

        protocol = self.protocol[key]
        cmap = colormaps[cmap_name]
        norm = Normalize(
            vmin=default_if_none(vmin, protocol.min()),
            vmax=default_if_none(vmax, protocol.max()),
        )

        t0 = pd.Timedelta(seconds=0)
        for t_end, val in protocol.items():
            t_end = cast(pd.Timedelta, t_end)
            ax.axvspan(
                t0.total_seconds(),
                t_end.total_seconds(),
                facecolor=cmap(norm(val)),
                edgecolor=None,
                alpha=alpha,
            )
            t0 = t_end  # type: ignore

        if add_legend:
            ax.add_artist(
                Legend(
                    ax,
                    handles=[
                        Patch(
                            facecolor=cmap(norm(val)),
                            alpha=alpha,
                            label=val,
                        )  # type: ignore
                        for val in protocol
                    ],
                    labels=protocol,
                    loc="lower left",
                    bbox_to_anchor=(1.0, 0.0),
                    title=key,
                )
            )


@dataclass
class McSteadyState:
    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def plot_violin(
        self,
        df: Literal["concs", "fluxes"],
        var: str,
        ax: Axis | None = None,
    ) -> Axis:
        _, ax = _fig_ax_if_neccessary(ax=ax)
        sns.violinplot(
            data=(self.concs if df == "concs" else self.fluxes)[var].unstack(),
            inner="point",
            split=True,
        )
        grid(ax)
        return ax


def _empty_conc_series(model: Model) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.compounds), fill_value=np.nan),
        index=model.compounds,
    )


def _empty_flux_series(model: Model) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.rates), fill_value=np.nan),
        index=model.rates,
    )


def _empty_conc_df(model: Model, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.compounds)),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.compounds,
    )


def _empty_flux_df(model: Model, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.rates)),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.rates,
    )


def empty_time_point(model: Model) -> tuple[pd.Series, pd.Series]:
    return _empty_conc_series(model), _empty_flux_series(model)


def empty_time_course(
    model: Model, time_points: Array
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _empty_conc_df(model, time_points), _empty_flux_df(model, time_points)


def _time_course_worker(
    pars: pd.Series,
    model: Model,
    y0: dict[str, float],
    time_points: Array,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c, v = (
        Simulator(model)
        .initialise(y0)
        .update_parameters(pars.to_dict())
        .simulate_and(time_points=time_points)
        .get_full_results_and_fluxes()
    )
    if c is None or v is None:
        return empty_time_course(model, time_points)
    return c, v


def time_course(
    model: Model,
    y0: dict[str, float],
    time_points: Array,
    mc_parameters: pd.DataFrame,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> McTimeCourse:
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
            _time_course_worker,
            model=model,
            y0=y0,
            time_points=time_points,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v[0].T for k, v in res.items()}
    fluxes = {k: v[1].T for k, v in res.items()}
    return McTimeCourse(
        concs=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
    )


def _time_course_over_protocol_worker(
    pars: pd.Series,
    model: Model,
    y0: dict[str, float],
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_points = np.linspace(
        0,
        protocol.index[-1].total_seconds(),
        len(protocol) * time_points_per_step,
    )
    s = Simulator(model).initialise(y0).update_parameters(pars.to_dict())
    for t_end, ser in protocol.iterrows():
        t_end = cast(pd.Timedelta, t_end)
        s.update_parameters(ser.to_dict())
        s.simulate(t_end.total_seconds(), steps=time_points_per_step)
    c, v = s.get_full_results_and_fluxes()
    if c is None or v is None:
        return empty_time_course(model, time_points)
    return c, v


def time_course_over_protocol(
    model: Model,
    y0: dict[str, float],
    protocol: pd.DataFrame,
    mc_parameters: pd.DataFrame,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> McTimeProtocol:
    res = parallelise(
        partial(
            _time_course_over_protocol_worker,
            model=model,
            y0=y0,
            protocol=protocol,
            time_points_per_step=time_points_per_step,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v[0].T for k, v in res.items()}
    fluxes = {k: v[1].T for k, v in res.items()}
    return McTimeProtocol(
        concs=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
        protocol=protocol,
    )


def _steady_state_scan_worker(
    pars: pd.Series,
    model: Model,
    y0: dict[str, float],
    parameter: str,
    parameter_values: Array,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        Simulator(model)
        .initialise(y0)
        .update_parameters(pars.to_dict())
        .parameter_scan_with_fluxes(
            parameter,
            parameter_values,
            multiprocessing=False,
            disable_tqdm=True,
        )
    )


def steady_state_scan(
    model: Model,
    y0: dict[str, float],
    parameter: str,
    parameter_values: Array,
    mc_parameters: pd.DataFrame,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> McSteadyState:
    res = parallelise(
        partial(
            _steady_state_scan_worker,
            model=model,
            y0=y0,
            parameter=parameter,
            parameter_values=parameter_values,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v[0].T for k, v in res.items()}
    fluxes = {k: v[1].T for k, v in res.items()}
    return McSteadyState(
        concs=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
    )
