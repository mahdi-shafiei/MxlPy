from __future__ import annotations

__all__ = [
    "_BaseRateSimulator",
    "_BaseSimulator",
]

import copy
import sys
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from modelbase2.types import Array, ArrayLike, Axes, Axis, Figure
from modelbase2.utils.plotting import (_get_plot_kwargs, _style_subplot, plot,
                                       plot_grid)

from . import (BASE_MODEL_TYPE, RATE_MODEL_TYPE, _AbstractRateModel,
               _AbstractStoichiometricModel)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from modelbase2.ode.integrators import AbstractIntegrator


def _empty_conc_series(model: _AbstractStoichiometricModel) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.compounds), fill_value=np.nan),
        index=model.compounds,
    )


def _empty_flux_series(model: _AbstractRateModel) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.rates), fill_value=np.nan),
        index=model.rates,
    )


def _empty_conc_df(
    model: _AbstractStoichiometricModel, time_points: Array
) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.compounds)),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.compounds,
    )


def _empty_flux_df(model: _AbstractRateModel, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.rates)),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.rates,
    )


def empty_time_point(model: _AbstractRateModel) -> tuple[pd.Series, pd.Series]:
    return _empty_conc_series(model), _empty_flux_series(model)


class _BaseSimulator(Generic[BASE_MODEL_TYPE], ABC):
    def __init__(
        self,
        model: BASE_MODEL_TYPE,
        integrator: type[AbstractIntegrator],
        y0: ArrayLike | None = None,
        results: list[pd.DataFrame] | None = None,
    ) -> None:
        self.model = model
        self._integrator = integrator
        self.integrator: AbstractIntegrator | None = None

        # For restoring purposes
        self.y0: ArrayLike | None = y0
        self.results: list[pd.DataFrame] | None = results

    def __reduce__(self) -> Any:
        """Pickle this class."""
        return (
            self.__class__,
            (
                self.model,
                self._integrator,
            ),
            (
                ("y0", self.y0),
                ("results", self.results),
            ),
        )

    def clear_results(self) -> None:
        """Clear simulation results."""
        self.results = None
        if self.integrator is not None:
            self.integrator.reset()

    def _initialise_integrator(self, *, y0: ArrayLike) -> None:
        """Initialise the integrator.

        Required for assimulo, as it needs y0 to initialise
        """
        self.integrator = self._integrator(
            rhs=self.model._get_rhs,  # noqa: SLF001
            y0=y0,
        )

    def get_integrator_params(self) -> dict[str, Any] | None:
        if self.integrator is None:
            return None
        return self.integrator.get_integrator_kwargs()

    @abstractmethod
    def copy(self) -> Any:
        """Create a copy."""

    def _normalise_split_results(
        self,
        *,
        results: list[pd.DataFrame],
        normalise: float | ArrayLike,
    ) -> list[pd.DataFrame]:
        if isinstance(normalise, int | float):
            return [i / normalise for i in results]
        if len(normalise) == len(results):
            return [(i.T / j).T for i, j in zip(results, normalise, strict=False)]

        results = []
        start = 0
        end = 0
        for i in results:
            end += len(i)
            results.append(i / np.reshape(normalise[start:end], (len(i), 1)))  # type: ignore
            start += end
        return results

    @abstractmethod
    def _test_run(self) -> None:
        """Perform a test step of the simulation in Python to get proper error handling."""

    def _save_simulation_results(
        self,
        *,
        results: pd.DataFrame,
        skipfirst: bool,
    ) -> None:
        if self.results is None:
            self.results = [results]
        elif skipfirst:
            self.results.append(results.iloc[1:, :])
        else:
            self.results.append(results)

    def simulate(
        self,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
        **integrator_kwargs: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Simulate the model."""
        if self.integrator is None:
            msg = "Initialise the simulator first."
            raise AttributeError(msg)

        if steps is not None and time_points is not None:
            warnings.warn(
                """
            You can either specify the steps or the time return points.
            I will use the time return points""",
                stacklevel=1,
            )
            if t_end is None:
                t_end = time_points[-1]
            time, results = self.integrator.integrate(
                t_end=t_end,
                time_points=time_points,
                **integrator_kwargs,  # type: ignore
            )
        elif time_points is not None:
            time, results = self.integrator.integrate(
                t_end=time_points[-1],
                time_points=time_points,
                **integrator_kwargs,  # type: ignore
            )
        elif steps is not None:
            if t_end is None:
                msg = "t_end must no be None"
                raise ValueError(msg)
            time, results = self.integrator.integrate(
                t_end=t_end,
                steps=steps,
                **integrator_kwargs,  # type: ignore
            )
        else:
            time, results = self.integrator.integrate(
                t_end=t_end,
                **integrator_kwargs,  # type: ignore
            )

        if time is None or results is None:
            return None

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_compounds()
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            results,
            index=time,
            columns=self.model.get_compounds(),
        )
        self._save_simulation_results(results=results_df, skipfirst=True)
        return results_df

    def simulate_and(
        self,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
        **integrator_kwargs: dict[str, Any],
    ) -> Self:
        self.simulate(
            t_end=t_end,
            steps=steps,
            time_points=time_points,
            **integrator_kwargs,
        )
        return self

    def simulate_to_steady_state(
        self,
        tolerance: float = 1e-6,
        simulation_kwargs: dict[str, Any] | None = None,
        *,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> pd.Series | None:
        """Simulate the model."""
        if self.integrator is None:
            msg = "Initialise the simulator first."
            raise AttributeError(msg)

        if simulation_kwargs is None:
            simulation_kwargs = {}

        time, results = self.integrator.integrate_to_steady_state(
            tolerance=tolerance,
            simulation_kwargs=simulation_kwargs,
            integrator_kwargs=integrator_kwargs,
            rel_norm=rel_norm,
        )
        if time is None or results is None:
            return None

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_compounds
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            data=[results],
            index=[time],
            columns=self.model.get_compounds(),
        )
        self._save_simulation_results(results=results_df, skipfirst=False)
        return results_df.iloc[-1]

    def simulate_to_steady_state_and(
        self,
        tolerance: float = 1e-6,
        simulation_kwargs: dict[str, Any] | None = None,
        *,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> Self:
        self.simulate_to_steady_state(
            tolerance=tolerance,
            simulation_kwargs=simulation_kwargs,
            rel_norm=rel_norm,
            **integrator_kwargs,
        )
        return self

    @overload
    def get_results(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> None | list[pd.DataFrame]: ...

    @overload
    def get_results(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> None | pd.DataFrame: ...

    @overload
    def get_results(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> None | pd.DataFrame: ...

    def get_results(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> None | pd.DataFrame | list[pd.DataFrame]:
        """Get simulation results."""
        if self.results is None:
            return None

        results = self.results.copy()
        if normalise is not None:
            results = self._normalise_split_results(
                results=results, normalise=normalise
            )
        if concatenated:
            return pd.concat(results, axis=0)

        return results

    def get_new_y0(self) -> dict[str, float] | None:
        if (res := self.get_results()) is None:
            return None
        return dict(res.iloc[-1])


class _BaseRateSimulator(Generic[RATE_MODEL_TYPE], _BaseSimulator[RATE_MODEL_TYPE]):  # type: ignore
    def __init__(
        self,
        model: RATE_MODEL_TYPE,
        integrator: type[AbstractIntegrator],
        y0: ArrayLike | None = None,
        results: list[pd.DataFrame] | None = None,
        parameters: list[dict[str, float]] | None = None,
    ) -> None:
        _BaseSimulator.__init__(
            self,
            model=model,
            integrator=integrator,
            y0=y0,
            results=results,
        )
        self.full_results: list[pd.DataFrame] | None = None
        self.fluxes: list[pd.DataFrame] | None = None
        self.simulation_parameters: list[dict[str, float]] | None = parameters

    def __reduce__(self) -> Any:
        """Pickle this class."""
        return (
            self.__class__,
            (
                self.model,
                self._integrator,
            ),
            (
                ("y0", self.y0),
                ("results", self.results),
                ("parameters", self.simulation_parameters),
            ),
        )

    def copy(self) -> Any:
        """Return a deepcopy of this class."""
        new = copy.deepcopy(self)
        if self.simulation_parameters is not None:
            new.simulation_parameters = self.simulation_parameters.copy()
        if self.fluxes is not None:
            new.fluxes = self.fluxes.copy()
        if self.full_results is not None:
            new.full_results = self.full_results.copy()
        if new.results is not None:
            new._initialise_integrator(y0=new.results[-1].to_numpy())  # noqa: SLF001
        elif new.y0 is not None:
            new.initialise(y0=new.y0, test_run=False)
        return new

    def clear_results(self) -> None:
        """Clear simulation results."""
        super().clear_results()
        self.full_results = None
        self.fluxes = None
        self.simulation_parameters = None

    def _test_run(self) -> None:
        """Test run of a single integration step to get proper error handling."""
        if self.y0 is not None:
            y = self.model.get_full_concentration_dict(y=self.y0, t=0)
            self.model.get_fluxes(y=y, t=0)
            self.model.get_right_hand_side(y=y, t=0)

    def initialise(
        self,
        y0: ArrayLike | dict[str, float],
        *,
        test_run: bool = True,
    ) -> Self:
        """Initialise the integrator."""
        if self.results is not None:
            self.clear_results()
        if isinstance(y0, dict):
            self.y0 = [y0[compound] for compound in self.model.get_compounds()]
        else:
            self.y0 = list(y0)
        self._initialise_integrator(y0=self.y0)

        if test_run:
            self._test_run()
        return self

    def update_parameter(
        self,
        parameter_name: str,
        parameter_value: float,
    ) -> Self:
        """Update a model parameter."""
        self.model.update_parameter(
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
        return self

    def scale_parameter(
        self,
        parameter_name: str,
        factor: float,
        *,
        verbose: bool = False,
    ) -> Self:
        """Scale a model parameter."""
        self.model.scale_parameter(
            parameter_name=parameter_name,
            factor=factor,
            verbose=verbose,
        )
        return self

    def update_parameters(self, parameters: dict[str, float]) -> Self:
        """Update model parameters."""
        self.model.update_parameters(parameters=parameters)
        return self

    def _save_simulation_results(
        self,
        *,
        results: pd.DataFrame,
        skipfirst: bool,
    ) -> None:
        super()._save_simulation_results(results=results, skipfirst=skipfirst)
        if self.simulation_parameters is None:
            self.simulation_parameters = []
        self.simulation_parameters.append(self.model.get_parameters())

    def simulate(
        self,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
        **integrator_kwargs: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Simulate the model.

        You can either supply only a terminal time point, or additionally also the
        number of steps or exact time points for which values should be returned.

        Parameters
        ----------
        t_end
            Last point of the integration
        steps
            Number of integration time steps to be returned
        time_points
            Explicit time points which shall be returned
        integrator_kwargs : dict
            Integrator options

        """
        results = super().simulate(
            t_end=t_end,
            steps=steps,
            time_points=time_points,
            **integrator_kwargs,
        )
        self.full_results = None
        self.fluxes = None
        return results

    def simulate_to_steady_state(
        self,
        tolerance: float = 1e-8,
        simulation_kwargs: dict[str, Any] | None = None,
        *,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> pd.Series | None:
        """Simulate the model to steady state."""
        results = super().simulate_to_steady_state(
            tolerance=tolerance,
            simulation_kwargs=simulation_kwargs,
            rel_norm=rel_norm,
            **integrator_kwargs,
        )
        self.full_results = None
        self.fluxes = None
        return results

    def _calculate_fluxes(self) -> None:
        if (fcd := self.get_full_results(concatenated=False)) is None:
            return
        if (pars := self.simulation_parameters) is None:
            return

        fluxes: list[pd.DataFrame] = []
        for y, p in zip(fcd, pars, strict=False):
            self.update_parameters(parameters=p)
            fluxes.append(self.model._get_fluxes_from_df(fcd=y))  # noqa: SLF001
        self.fluxes = fluxes

    @overload
    def get_fluxes(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame | None: ...

    @overload
    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | None: ...

    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get the model fluxes for the simulation."""
        if self.results is None:
            return None
        if self.fluxes is None:
            self._calculate_fluxes()

        fluxes = self.fluxes
        if fluxes is None:
            return None
        if normalise is not None:
            fluxes = self._normalise_split_results(results=fluxes, normalise=normalise)
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    def _calculate_full_results(
        self,
        *,
        include_readouts: bool,
    ) -> None:
        all_full_results: list[pd.DataFrame] = []
        if (results := self.results) is None:
            raise ValueError
        if (params := self.simulation_parameters) is None:
            raise ValueError

        for res, p in zip(results, params, strict=False):
            self.update_parameters(parameters=p)
            full_results = self.model.get_full_concentration_dict(
                y=res.to_numpy(),
                t=res.index.to_numpy(),
                include_readouts=include_readouts,
            )
            del full_results["time"]
            all_full_results.append(pd.DataFrame(data=full_results, index=res.index))
        self.full_results = all_full_results

    @overload
    def get_full_results(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
        include_readouts: bool = True,
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_full_results(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
        include_readouts: bool = True,
    ) -> pd.DataFrame | None: ...

    @overload
    def get_full_results(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
    ) -> pd.DataFrame | None: ...

    def get_full_results(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get simulation results and derived compounds."""
        if self.full_results is None:
            self._calculate_full_results(include_readouts=include_readouts)

        if (full_results := self.full_results) is None:
            return None

        full_results = full_results.copy()

        if normalise is not None:
            full_results = self._normalise_split_results(
                results=full_results,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(full_results, axis=0)
        return full_results

    def get_results_and_fluxes(
        self,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        return self.get_results(), self.get_fluxes()

    def get_full_results_and_fluxes(
        self,
        *,
        include_readouts: bool = True,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        return (
            self.get_full_results(include_readouts=include_readouts),
            self.get_fluxes(),
        )

    def get_right_hand_side(
        self,
        *,
        annotate_names: bool = True,
    ) -> pd.DataFrame | None:
        rhs = pd.DataFrame()

        if (res := self.results) is None:
            return None
        if (pars := self.simulation_parameters) is None:
            return None

        for y, p in zip(res, pars, strict=False):
            self.update_parameters(p)
            rhs = pd.concat(
                (
                    rhs,
                    pd.DataFrame(
                        {
                            ti: self.model.get_right_hand_side(
                                y=yi.to_numpy(),
                                t=cast(float, ti),
                                annotate_names=annotate_names,
                            )
                            for ti, yi in y.iterrows()
                        }
                    ),
                )
            )
        return rhs

    @staticmethod
    def _parameter_scan_worker(
        parameter_value: float,
        *,
        parameter_name: str,
        model: RATE_MODEL_TYPE,
        sim: type[_BaseRateSimulator],
        integrator: type[AbstractIntegrator],
        tolerance: float,
        y0: ArrayLike,
        integrator_kwargs: dict[str, Any],
        include_fluxes: bool,
        rel_norm: bool,
    ) -> tuple[float, pd.Series, pd.Series]:
        m = model.copy()
        s = sim(model=m, integrator=integrator)
        s.initialise(y0=y0, test_run=False)
        s.update_parameter(
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )

        if (
            s.simulate_to_steady_state(
                tolerance=tolerance,
                rel_norm=rel_norm,
                **integrator_kwargs,
            )
            is None
        ):
            return parameter_value, _empty_conc_series(model), _empty_flux_series(model)

        if (full_results := s.get_full_results(concatenated=True)) is None:
            return parameter_value, _empty_conc_series(model), _empty_flux_series(model)
        last_full_results = full_results.iloc[-1]

        if include_fluxes:
            if (fluxes := s.get_fluxes(concatenated=True)) is None:
                return (
                    parameter_value,
                    _empty_conc_series(model),
                    _empty_flux_series(model),
                )
            last_fluxes = fluxes.iloc[-1]
        else:
            last_fluxes = pd.Series()

        return parameter_value, last_full_results, last_fluxes

    def parameter_scan(
        self,
        parameter_name: str,
        parameter_values: ArrayLike,
        tolerance: float = 1e-8,
        *,
        multiprocessing: bool = True,
        max_workers: int | None = None,
        disable_tqdm: bool = False,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """Scan the model steady state changes caused by a change to a parameter."""
        return self.parameter_scan_with_fluxes(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            tolerance=tolerance,
            multiprocessing=multiprocessing,
            max_workers=max_workers,
            disable_tqdm=disable_tqdm,
            rel_norm=rel_norm,
            **integrator_kwargs,
        )[0]

    def parameter_scan_with_fluxes(
        self,
        parameter_name: str,
        parameter_values: ArrayLike,
        tolerance: float = 1e-8,
        *,
        multiprocessing: bool = True,
        disable_tqdm: bool = False,
        max_workers: int | None = None,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scan the model steady state changes caused by a change to a parameter."""
        if sys.platform in ["win32", "cygwin"]:
            warnings.warn(
                """
                Windows does not behave well with multiple processes.
                Falling back to threading routine.""",
                stacklevel=1,
            )
        worker = partial(
            self._parameter_scan_worker,
            parameter_name=parameter_name,
            model=self.model,
            sim=self.__class__,
            integrator=self._integrator,
            tolerance=tolerance,
            y0=self.y0,
            integrator_kwargs=integrator_kwargs,
            include_fluxes=True,
            rel_norm=rel_norm,
        )

        results: Iterable[tuple[float, pd.Series, pd.Series]]
        if sys.platform in ["win32", "cygwin"] or not multiprocessing:
            results = tqdm(
                map(worker, parameter_values),
                total=len(parameter_values),
                desc=parameter_name,
                disable=disable_tqdm,
            )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pe:
                results = tqdm(pe.map(worker, parameter_values))
        concentrations = {}
        fluxes = {}
        for i, conc, flux in results:
            concentrations[i] = conc
            fluxes[i] = flux
        return (
            pd.DataFrame(concentrations).T,
            pd.DataFrame(fluxes).T,
        )

    def parameter_scan_2d(
        self,
        p1: tuple[str, ArrayLike],
        p2: tuple[str, ArrayLike],
        tolerance: float = 1e-8,
        *,
        disable_tqdm: bool = False,
        multiprocessing: bool = True,
        max_workers: int | None = None,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> dict[float, pd.DataFrame]:
        cs = {}
        parameter_name1, parameter_values1 = p1
        parameter_name2, parameter_values2 = p2
        original_pars = self.model.get_parameters().copy()
        for value in tqdm(
            parameter_values2, total=len(parameter_values2), desc=parameter_name2
        ):
            self.update_parameter(parameter_name2, value)
            cs[value] = self.parameter_scan(
                parameter_name1,
                parameter_values1,
                tolerance=tolerance,
                disable_tqdm=disable_tqdm,
                multiprocessing=multiprocessing,
                max_workers=max_workers,
                rel_norm=rel_norm,
                **integrator_kwargs,
            )
        self.update_parameters(original_pars)
        return cs

    def parameter_scan_2d_with_fluxes(
        self,
        p1: tuple[str, ArrayLike],
        p2: tuple[str, ArrayLike],
        tolerance: float = 1e-8,
        *,
        disable_tqdm: bool = False,
        multiprocessing: bool = True,
        max_workers: int | None = None,
        rel_norm: bool = False,
        **integrator_kwargs: dict[str, Any],
    ) -> tuple[dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
        cs = {}
        vs = {}
        parameter_name1, parameter_values1 = p1
        parameter_name2, parameter_values2 = p2
        original_pars = self.model.get_parameters().copy()
        for value in tqdm(
            parameter_values2, total=len(parameter_values2), desc=parameter_name2
        ):
            self.update_parameter(parameter_name2, value)
            c, v = self.parameter_scan_with_fluxes(
                parameter_name1,
                parameter_values1,
                tolerance=tolerance,
                multiprocessing=multiprocessing,
                disable_tqdm=disable_tqdm,
                max_workers=max_workers,
                rel_norm=rel_norm,
                **integrator_kwargs,  # type: ignore
            )
            cs[value] = c
            vs[value] = v
        self.update_parameters(original_pars)
        return cs, vs

    def plot_log(
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        compounds = self.model.get_compounds()
        y = cast(
            pd.DataFrame,
            self.get_full_results(normalise=normalise, concatenated=True),
        )
        if y is None:
            return None, None
        fig, ax = plot(
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
        ax.set_xscale("log")
        ax.set_yscale("log")
        return fig, ax

    def plot_semilog(
        self,
        *,
        log_axis: str = "y",
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        compounds = self.model.get_compounds()
        y = cast(
            pd.DataFrame,
            self.get_full_results(normalise=normalise, concatenated=True),
        )
        if y is None:
            return None, None
        fig, ax = plot(
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
        if log_axis == "y":
            ax.set_yscale("log")
        elif log_axis == "x":
            ax.set_xscale("log")
        else:
            msg = "log_axis must be either x or y"
            raise ValueError(msg)
        return fig, ax

    def plot_derived(
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        compounds = self.model.get_derived_compounds()
        y = cast(
            pd.DataFrame,
            self.get_full_results(normalise=normalise, concatenated=True),
        )
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

    def plot_all(
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        compounds = self.model.get_all_compounds()
        y = cast(
            pd.DataFrame,
            self.get_full_results(normalise=normalise, concatenated=True),
        )
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

    def plot_selection(
        self,
        *,
        compounds: list[str],
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        y = cast(
            pd.DataFrame,
            self.get_full_results(normalise=normalise, concatenated=True),
        )
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

    def plot_grid(
        self,
        compound_groups: list[list[str]],
        *,
        ncols: int | None = None,
        sharex: bool = True,
        sharey: bool = True,
        xlabels: str | Iterable[str] | None = None,
        ylabels: str | Iterable[str] | None = None,
        normalise: float | ArrayLike | None = None,
        plot_titles: Iterable[str] | None = None,
        figure_title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axes | None]:
        """Plot simulation results of the compound groups as a grid.

        Examples
        --------
        >>> plot_grid([["x1", "x2"], ["x3", "x4]])

        """
        y = cast(
            pd.DataFrame,
            self.get_full_results(normalise=normalise, concatenated=True),
        )
        if y is None:
            return None, None
        plot_groups = [(y.loc[:, compounds],) for compounds in compound_groups]
        return plot_grid(
            plot_groups=plot_groups,  # type: ignore
            legend_groups=compound_groups,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            xlabels=xlabels,
            ylabels=ylabels,
            figure_title=figure_title,
            plot_titles=plot_titles,
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

    def plot_derivatives(
        self,
        compounds: list[str],
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        rhs = self.get_right_hand_side(annotate_names=False)
        if len(rhs) == 0:
            return None, None

        return plot(
            plot_args=(rhs.loc[:, compounds],),
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

    def plot_against_variable(
        self,
        variable: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if xlabel is None:
            xlabel = variable
        results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
        if results is None:
            return None, None
        compounds = cast(list[str], self.model.get_compounds())
        x = results.loc[:, variable].values  # type: ignore
        y = results.loc[:, compounds].values
        return plot(
            plot_args=(x, y),
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

    def plot_derived_against_variable(
        self,
        variable: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if xlabel is None:
            xlabel = variable
        results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
        if results is None:
            return None, None
        compounds = cast(list[str], self.model.get_derived_compounds())
        x = results.loc[:, variable].values  # type: ignore
        y = results.loc[:, compounds].values
        return plot(
            plot_args=(x, y),
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

    def plot_all_against_variable(
        self,
        variable: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if xlabel is None:
            xlabel = variable
        results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
        if results is None:
            return None, None
        compounds = cast(list[str], self.model.get_all_compounds())
        x = results.loc[:, variable].values  # type: ignore
        y = results.loc[:, compounds].values
        return plot(
            plot_args=(x, y),
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

    def plot_selection_against_variable(
        self,
        compounds: Iterable[str],
        variable: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if xlabel is None:
            xlabel = variable
        results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
        if results is None:
            return None, None
        x = results.loc[:, variable].values  # type: ignore
        y = results.loc[:, compounds].values  # type: ignore
        return plot(
            plot_args=(x, y),
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

    def plot_fluxes(
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        y = self.get_fluxes(normalise=normalise, concatenated=True)
        if y is None:
            return None, None
        return plot(
            plot_args=(y,),
            legend=y.columns,
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

    def plot_flux_selection(
        self,
        rate_names: list[str],
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
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        y = self.get_fluxes(normalise=normalise, concatenated=True)
        if y is None:
            return None, None
        y = cast(pd.DataFrame, y)
        return plot(
            plot_args=(y.loc[:, rate_names],),
            legend=rate_names,
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

    def plot_fluxes_grid(
        self,
        rate_groups: list[list[str]],
        *,
        ncols: int | None = None,
        sharex: bool = True,
        sharey: bool = True,
        xlabels: list[str] | None = None,
        ylabels: list[str] | None = None,
        normalise: float | ArrayLike | None = None,
        plot_titles: Iterable[str] | None = None,
        figure_title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Array | None]:
        """Plot simulation results of the compound groups as a grid.

        Examples
        --------
        >>> plot_fluxes_grid([["v1", "v2"], ["v3", "v4]])

        """
        fluxes = self.get_fluxes(normalise=normalise, concatenated=True)
        if fluxes is None:
            return None, None
        fluxes = cast(pd.DataFrame, fluxes)
        plot_groups = [(cast(Array, fluxes.loc[:, group]),) for group in rate_groups]
        return plot_grid(
            plot_groups=plot_groups,  # type: ignore
            legend_groups=rate_groups,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            xlabels=xlabels,
            ylabels=ylabels,
            figure_title=figure_title,
            plot_titles=plot_titles,
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

    def plot_fluxes_against_variable(
        self,
        variable: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if (c := self.get_results()) is None:
            return None, None
        if (v := self.get_fluxes(concatenated=True)) is None:
            return None, None
        rate_names = self.model.get_rate_names()
        x = c.loc[:, variable]
        y = v.loc[:, rate_names].to_numpy()
        return plot(
            plot_args=(x, y),
            legend=rate_names,
            xlabel=variable if xlabel is None else xlabel,
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

    def plot_flux_selection_against_variable(
        self,
        rate_names: list[str],
        variable: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if (c := self.get_results()) is None:
            return None, None
        if (v := self.get_fluxes()) is None:
            return None, None
        x = c.loc[:, variable]
        y = v.loc[:, rate_names].to_numpy()
        return plot(
            plot_args=(x, y),
            legend=rate_names,
            xlabel=variable if xlabel is None else xlabel,
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

    def plot_phase_plane(
        self,
        cpd1: str,
        cpd2: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        if (c := self.get_results()) is None:
            return None, None

        return plot(
            plot_args=(c.loc[:, cpd1], c.loc[:, cpd2]),
            legend=None,
            xlabel=cpd1 if xlabel is None else xlabel,
            ylabel=cpd2 if ylabel is None else ylabel,
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

    def plot_phase_space(
        self,
        cpd1: str,
        cpd2: str,
        cpd3: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        kwargs = _get_plot_kwargs(
            figure_kwargs=figure_kwargs,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            grid_kwargs=grid_kwargs,
            tick_kwargs=tick_kwargs,
            label_kwargs=label_kwargs,
            title_kwargs=title_kwargs,
            legend_kwargs=legend_kwargs,
        )
        kwargs["subplot"].update({"projection": "3d"})

        if (c := self.get_results()) is None:
            return None, None

        x = c.loc[:, cpd1]
        y = c.loc[:, cpd2]
        z = c.loc[:, cpd3]

        if x is None or y is None or z is None:
            return None, None

        xlabel = cpd1 if xlabel is None else xlabel
        ylabel = cpd2 if ylabel is None else ylabel
        zlabel = cpd3 if zlabel is None else zlabel

        if ax is None:
            fig, ax = plt.subplots(
                1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"]
            )
        else:
            fig = ax.get_figure()
        fig = cast(Figure, fig)
        ax = cast(Axis, ax)

        ax.plot(
            x,
            y,
            z,
            **kwargs["plot"],
        )
        _style_subplot(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            title=title,
            grid=grid,
            kwargs=kwargs,
        )
        if tight_layout:
            fig.tight_layout()
        return fig, ax

    def plot_trajectories(
        self,
        cpd1: str,
        cpd2: str,
        *,
        cpd1_bounds: tuple[float, float],
        cpd2_bounds: tuple[float, float],
        n: int,
        y0: dict[str, float],
        t0: float = 0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        xlabel = cpd1 if xlabel is None else xlabel
        ylabel = cpd2 if ylabel is None else ylabel

        kwargs = _get_plot_kwargs(
            figure_kwargs=figure_kwargs,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            grid_kwargs=grid_kwargs,
            tick_kwargs=tick_kwargs,
            label_kwargs=label_kwargs,
            title_kwargs=title_kwargs,
            legend_kwargs=legend_kwargs,
        )

        x = np.linspace(*cpd1_bounds, n)
        y = np.linspace(*cpd2_bounds, n)
        u = np.zeros((n, n))
        v = np.zeros((n, n))

        fcd = self.model.get_full_concentration_dict(y=y0, t=t0)
        for i, s1 in enumerate(x):
            for j, s2 in enumerate(y):
                # Update y0 to new values
                fcd.update({cpd1: s1, cpd2: s2})
                rhs = self.model.get_right_hand_side(y=fcd, t=t0)
                u[i, j] = rhs[f"d{cpd1}dt"]
                v[i, j] = rhs[f"d{cpd2}dt"]

        if ax is None:
            fig, ax = plt.subplots(
                1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"]
            )
        else:
            fig = ax.get_figure()
        fig = cast(Figure, fig)
        ax = cast(Axis, ax)

        ax.quiver(x, y, u.T, v.T)
        _style_subplot(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            grid=grid,
            kwargs=kwargs,
        )
        if tight_layout:
            fig.tight_layout()
        return fig, ax

    def plot_3d_trajectories(
        self,
        cpd1: str,
        cpd2: str,
        cpd3: str,
        *,
        cpd1_bounds: tuple[float, float],
        cpd2_bounds: tuple[float, float],
        cpd3_bounds: tuple[float, float],
        n: int,
        y0: dict[str, float],
        t0: float = 0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        tight_layout: bool = True,
        ax: Axis | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        grid_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        label_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axis | None]:
        kwargs = _get_plot_kwargs(
            figure_kwargs=figure_kwargs,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            grid_kwargs=grid_kwargs,
            tick_kwargs=tick_kwargs,
            label_kwargs=label_kwargs,
            title_kwargs=title_kwargs,
            legend_kwargs=legend_kwargs,
        )
        kwargs["subplot"].update({"projection": "3d"})

        x = np.linspace(*cpd1_bounds, n)
        y = np.linspace(*cpd2_bounds, n)
        z = np.linspace(*cpd3_bounds, n)
        u = np.zeros((n, n, n))
        v = np.zeros((n, n, n))
        w = np.zeros((n, n, n))

        fcd = self.model.get_full_concentration_dict(y=y0, t=t0)
        for i, s1 in enumerate(x):
            for j, s2 in enumerate(y):
                for k, s3 in enumerate(y):
                    fcd.update({cpd1: s1, cpd2: s2, cpd3: s3})
                    rhs = self.model.get_right_hand_side(y=fcd, t=t0)
                    u[i, j, k] = rhs[f"d{cpd1}dt"]
                    v[i, j, k] = rhs[f"d{cpd2}dt"]
                    w[i, j, k] = rhs[f"d{cpd3}dt"]

        if ax is None:
            fig, ax = plt.subplots(
                1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"]
            )
        else:
            fig = ax.get_figure()
        fig = cast(Figure, fig)
        ax = cast(Axis, ax)

        X, Y, Z = np.meshgrid(x, y, z)
        ax.quiver(
            X,
            Y,
            Z,
            np.transpose(u, [1, 0, 2]),
            np.transpose(v, [1, 0, 2]),
            np.transpose(w, [1, 0, 2]),
            length=0.05,
            normalize=True,
            alpha=0.5,
        )
        xlabel = cpd1 if xlabel is None else xlabel
        ylabel = cpd2 if ylabel is None else ylabel
        zlabel = cpd3 if zlabel is None else zlabel
        _style_subplot(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            title=title,
            grid=grid,
            kwargs=kwargs,
        )
        if tight_layout:
            fig.tight_layout()
        return fig, ax
