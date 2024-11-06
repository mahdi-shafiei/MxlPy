from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, cast, overload

import numpy as np
import pandas as pd

from modelbase2.integrators import DefaultIntegrator

if TYPE_CHECKING:
    from modelbase2.types import ArrayLike, IntegratorProtocol, ModelProtocol


def _normalise_split_results(
    results: list[pd.DataFrame],
    normalise: float | ArrayLike,
) -> list[pd.DataFrame]:
    if isinstance(normalise, int | float):
        return [i / normalise for i in results]
    if len(normalise) == len(results):
        return [(i.T / j).T for i, j in zip(results, normalise, strict=True)]

    results = []
    start = 0
    end = 0
    for i in results:
        end += len(i)
        results.append(i / np.reshape(normalise[start:end], (len(i), 1)))  # type: ignore
        start += end
    return results


@dataclass(
    init=False,
    slots=True,
    eq=False,
)
class Simulator:
    model: ModelProtocol
    y0: ArrayLike
    integrator: IntegratorProtocol
    concs: list[pd.DataFrame] | None
    args: list[pd.DataFrame] | None
    simulation_parameters: list[dict[str, float]] | None

    def __init__(
        self,
        model: ModelProtocol,
        y0: dict[str, float] | None = None,
        integrator: type[IntegratorProtocol] = DefaultIntegrator,
        *,
        test_run: bool = True,
    ) -> None:
        self.model = model
        if y0 is None:
            self.y0 = model.get_initial_conditions()
        else:
            self.y0 = [y0[k] for k in model.get_variable_names()]

        self.integrator = integrator(
            self.model._get_rhs,  # noqa: SLF001
            y0=self.y0,
        )
        self.concs = None
        self.args = None
        self.simulation_parameters = None

        if test_run:
            y0 = dict(zip(model.get_variable_names(), self.y0, strict=True))
            self.model.get_full_concs(y0, 0)
            self.model.get_fluxes(y0, 0)
            self.model.get_right_hand_side(y0, 0)

    def _save_simulation_results(
        self,
        *,
        results: pd.DataFrame,
        skipfirst: bool,
    ) -> None:
        if self.concs is None:
            self.concs = [results]
        elif skipfirst:
            self.concs.append(results.iloc[1:, :])
        else:
            self.concs.append(results)

        if self.simulation_parameters is None:
            self.simulation_parameters = []
        self.simulation_parameters.append(self.model.parameters)

    def clear_results(self) -> None:
        """Clear simulation results."""
        self.concs = None
        self.args = None
        self.simulation_parameters = None
        if self.integrator is not None:
            self.integrator.reset()

    def simulate(
        self,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> Self:
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
        """
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
            )
        elif time_points is not None:
            time, results = self.integrator.integrate(
                t_end=time_points[-1],
                time_points=time_points,
            )
        elif steps is not None:
            if t_end is None:
                msg = "t_end must no be None"
                raise ValueError(msg)
            time, results = self.integrator.integrate(
                t_end=t_end,
                steps=steps,
            )
        else:
            time, results = self.integrator.integrate(
                t_end=t_end,
            )

        if time is None or results is None:
            return self

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_compounds()
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            results,
            index=time,
            columns=self.model.get_variable_names(),
        )
        self._save_simulation_results(results=results_df, skipfirst=True)
        return self

    def simulate_to_steady_state(
        self,
        tolerance: float = 1e-6,
        *,
        rel_norm: bool = False,
    ) -> Self:
        time, results = self.integrator.integrate_to_steady_state(
            tolerance=tolerance,
            rel_norm=rel_norm,
        )
        if time is None or results is None:
            return self

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_compounds
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            data=[results],
            index=[time],
            columns=self.model.get_variable_names(),
        )
        self._save_simulation_results(results=results_df, skipfirst=False)
        return self

    def simulate_over_protocol(
        self,
        protocol: pd.DataFrame,
        time_points_per_step: int = 10,
    ) -> Self:
        for t_end, pars in protocol.iterrows():
            t_end = cast(pd.Timedelta, t_end)
            self.model.update_parameters(pars.to_dict())
            self.simulate(t_end.total_seconds(), steps=time_points_per_step)
        return self

    def _get_args_vectorised(
        self,
        concs: list[pd.DataFrame],
        params: list[dict[str, float]],
        *,
        include_readouts: bool = True,
    ) -> list[pd.DataFrame]:
        args: list[pd.DataFrame] = []

        for res, p in zip(concs, params, strict=True):
            self.model.update_parameters(p)
            args.append(
                self.model._get_args_vectorised(  # noqa: SLF001
                    concs=res,
                    include_readouts=include_readouts,
                )
            )
        return args

    @overload
    def get_concs(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> None | list[pd.DataFrame]: ...

    @overload
    def get_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> None | pd.DataFrame: ...

    @overload
    def get_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True] = True,
    ) -> None | pd.DataFrame: ...

    def get_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> None | pd.DataFrame | list[pd.DataFrame]:
        """Get simulation results."""
        if self.concs is None:
            return None

        results = self.concs.copy()
        if normalise is not None:
            results = _normalise_split_results(results=results, normalise=normalise)
        if concatenated:
            return pd.concat(results, axis=0)

        return results

    @overload
    def get_full_concs(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
        include_readouts: bool = True,
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
        include_readouts: bool = True,
    ) -> pd.DataFrame | None: ...

    @overload
    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
    ) -> pd.DataFrame | None: ...

    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get simulation results and derived compounds."""
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None
        if (args := self.args) is None:
            args = self._get_args_vectorised(concs, params)

        names = (
            self.model.get_variable_names() + self.model.get_derived_variable_names()
        )
        if include_readouts:
            names.extend(self.model.get_readout_names())
        full_concs = [i.loc[:, names] for i in args]
        if normalise is not None:
            full_concs = _normalise_split_results(
                results=full_concs,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(full_concs, axis=0)
        return full_concs

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
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None
        if (args := self.args) is None:
            args = self._get_args_vectorised(concs, params)

        fluxes: list[pd.DataFrame] = []
        for y, p in zip(args, params, strict=True):
            self.model.update_parameters(p)
            fluxes.append(self.model._get_fluxes_vectorised(args=y))  # noqa: SLF001

        if normalise is not None:
            fluxes = _normalise_split_results(
                results=fluxes,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    def get_concs_and_fluxes(
        self,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        return self.get_concs(), self.get_fluxes()

    def get_full_concs_and_fluxes(
        self,
        *,
        include_readouts: bool = True,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        return (
            self.get_full_concs(include_readouts=include_readouts),
            self.get_fluxes(),
        )

    def get_new_y0(self) -> dict[str, float] | None:
        if (res := self.get_concs()) is None:
            return None
        return dict(res.iloc[-1])
