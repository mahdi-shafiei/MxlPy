from __future__ import annotations

__all__ = [
    "_Simulate",
]

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
from scipy.optimize import minimize

from modelbase2.core.utils import warning_on_one_line
from modelbase2.ode.models import Model
from modelbase2.utils.plotting import plot, plot_grid

from . import _BaseRateSimulator

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure

    from modelbase2.ode.integrators import AbstractIntegrator
    from modelbase2.typing import Array, ArrayLike, Axes, Axis

warnings.formatwarning = warning_on_one_line  # type: ignore


def _get_file_type_from_path(path: Path, filetype: str | None) -> str:
    if filetype is None:
        file_type = "json" if not path.suffix else path.suffix[1:]
    else:
        file_type = filetype
    return file_type


def _add_suffix(path: Path, filetype: str) -> Path:
    if not path.suffix:
        path = path.parent / (path.name + f".{filetype}")
    return path


class _Simulate(_BaseRateSimulator[Model]):
    """Simulator for ODE models."""

    def __init__(
        self,
        model: Model,
        integrator: type[AbstractIntegrator],
        y0: ArrayLike | None = None,
        results: list[pd.DataFrame] | None = None,
        parameters: list[dict[str, float]] | None = None,
    ) -> None:
        """Parameters
        ----------
        kwargs
            {parameters}

        """
        super().__init__(
            model=model,
            integrator=integrator,
            y0=y0,
            results=results,
            parameters=parameters,
        )

    def fit_steady_state(
        self,
        p0: dict[str, float],
        data: ArrayLike,
    ) -> dict[str, float]:
        par_names = list(p0.keys())
        x0 = list(p0.values())
        p_orig = self.model.get_parameters().copy()

        def residual(par_values: ArrayLike, data: ArrayLike) -> float:
            self.clear_results()
            self.update_parameters(dict(zip(par_names, par_values, strict=False)))
            y_ss = self.simulate_to_steady_state()
            if y_ss is None:
                return cast(float, np.inf)
            return cast(float, np.sqrt(np.mean(np.square(data - y_ss.to_numpy()))))

        res = dict(
            zip(
                par_names,
                minimize(
                    residual,
                    x0=x0,
                    args=(data,),
                    bounds=[(1e-12, 1e6) for _ in range(len(p0))],
                    method="L-BFGS-B",
                ).x,
                strict=False,
            )
        )
        self.model.update_parameters(p_orig)
        return res

    def fit_time_series(
        self,
        p0: dict[str, float],
        data: ArrayLike,
        time_points: ArrayLike,
    ) -> dict[str, float]:
        par_names = list(p0.keys())
        x0 = list(p0.values())
        p_orig = self.model.get_parameters().copy()
        if len(data) != len(time_points):
            raise ValueError

        def residual(
            par_values: ArrayLike, data: ArrayLike, time_points: ArrayLike
        ) -> float:
            self.clear_results()
            self.update_parameters(dict(zip(par_names, par_values, strict=False)))
            if (y := self.simulate(time_points=time_points)) is None:
                return cast(float, np.inf)
            return cast(float, np.sqrt(np.mean(np.square(data - y.to_numpy()))))

        res = dict(
            zip(
                par_names,
                minimize(
                    residual,
                    x0=x0,
                    args=(data, time_points),
                    bounds=[(1e-12, 1e6) for _ in range(len(p0))],
                    method="L-BFGS-B",
                ).x,
                strict=False,
            )
        )
        self.model.update_parameters(p_orig)
        return res

    def plot(
        self,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        normalise: float | ArrayLike | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any | None] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        """Plot simulation results for a selection of compounds."""
        compounds = self.model.get_compounds()
        y = self.get_full_results(normalise=normalise, concatenated=True)
        if y is None:
            return None, None
        return plot(
            plot_args=(y.loc[:, compounds],),
            legend=compounds,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            grid=grid,
            tight_layout=tight_layout,
            ax=ax,
            figure_kwargs=figure_kwargs,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_kwargs=tick_kwargs,
            label_kwargs=label_kwargs,
            title_kwargs=title_kwargs,
        )

    def plot_producing_and_consuming(
        self,
        compound: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any | None] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[None, None] | tuple[Figure, Axes]:
        producing: list[Array] = []
        consuming: list[Array] = []
        producing_names: list[str] = []
        consuming_names: list[str] = []

        if (v := self.get_fluxes()) is None:
            return None, None

        title = compound if title is None else title

        for rate_name, factor in self.model.stoichiometries_by_compounds[
            compound
        ].items():
            if factor > 0:
                producing.append(v[rate_name].to_numpy() * factor)  # type: ignore
                producing_names.append(rate_name)
            else:
                consuming.append(v[rate_name].to_numpy() * -factor)  # type: ignore
                consuming_names.append(rate_name)

        return plot_grid(
            plot_groups=[
                (v.index.to_numpy(), np.array(producing).T),
                (v.index.to_numpy(), np.array(consuming).T),
            ],
            legend_groups=[producing_names, consuming_names],
            xlabels=xlabel,
            ylabels=ylabel,
            figure_title=title,
            plot_titles=("Producing", "Consuming"),
            grid=grid,
            tight_layout=tight_layout,
            figure_kwargs=figure_kwargs,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_kwargs=tick_kwargs,
            label_kwargs=label_kwargs,
            title_kwargs=title_kwargs,
        )
