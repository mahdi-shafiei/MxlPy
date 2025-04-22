"""Simulation Module.

This module provides classes and functions for simulating metabolic models.
It includes functionality for running simulations, normalizing results, and
retrieving simulation data.

Classes:
    Simulator: Class for running simulations on a metabolic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, cast, overload

import numpy as np
import pandas as pd

from modelbase2.integrators import DefaultIntegrator

__all__ = ["Simulator"]

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelbase2.model import Model
    from modelbase2.types import ArrayLike, IntegratorProtocol


def _normalise_split_results(
    results: list[pd.DataFrame],
    normalise: float | ArrayLike,
) -> list[pd.DataFrame]:
    """Normalize split results by a given factor or array.

    Args:
        results: List of DataFrames containing the results to normalize.
        normalise: Normalization factor or array.

    Returns:
        list[pd.DataFrame]: List of normalized DataFrames.

    """
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
    """Simulator class for running simulations on a metabolic model.

    Attributes:
        model: Model instance to simulate.
        y0: Initial conditions for the simulation.
        integrator: Integrator protocol to use for the simulation.
        concs: List of DataFrames containing concentration results.
        dependent: List of DataFrames containing argument values.
        simulation_parameters: List of dictionaries containing simulation parameters.

    """

    model: Model
    y0: ArrayLike
    integrator: IntegratorProtocol
    concs: list[pd.DataFrame] | None
    dependent: list[pd.DataFrame] | None
    simulation_parameters: list[dict[str, float]] | None

    def __init__(
        self,
        model: Model,
        y0: dict[str, float] | None = None,
        integrator: Callable[
            [Callable, ArrayLike], IntegratorProtocol
        ] = DefaultIntegrator,
        *,
        test_run: bool = True,
    ) -> None:
        """Initialize the Simulator.

        Args:
            model (Model): The model to be simulated.
            y0 (dict[str, float] | None, optional): Initial conditions for the model variables.
                If None, the initial conditions are obtained from the model. Defaults to None.
            integrator (Callable[[Callable, ArrayLike], IntegratorProtocol], optional): The integrator to use for the simulation.
                Defaults to DefaultIntegrator.
            test_run (bool, optional): If True, performs a test run to ensure the model's methods
                (get_full_concs, get_fluxes, get_right_hand_side) work correctly with the initial conditions.
                Defaults to True.

        """
        self.model = model
        y0 = model.get_initial_conditions() if y0 is None else y0
        self.y0 = [y0[k] for k in model.get_variable_names()]

        self.integrator = integrator(self.model, self.y0)
        self.concs = None
        self.simulation_parameters = None

        if test_run:
            y0 = dict(zip(model.get_variable_names(), self.y0, strict=True))
            self.model.get_right_hand_side(y0, 0)

    def _save_simulation_results(
        self,
        *,
        results: pd.DataFrame,
        skipfirst: bool,
    ) -> None:
        """Save simulation results.

        Args:
            results: DataFrame containing the simulation results.
            skipfirst: Whether to skip the first row of results.

        """
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
        self.dependent = None
        self.simulation_parameters = None
        if self.integrator is not None:
            self.integrator.reset()

    def _handle_simulation_results(
        self,
        time: ArrayLike | None,
        results: ArrayLike | None,
        *,
        skipfirst: bool,
    ) -> None:
        """Handle simulation results.

        Args:
            time: Array of time points for the simulation.
            results: Array of results for the simulation.
            skipfirst: Whether to skip the first row of results.

        """
        if time is None or results is None:
            # Need to clear results in case continued integration fails
            # to keep expectation that failure = None
            self.clear_results()
            return

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_variable_names()
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            results,
            index=time,
            columns=self.model.get_variable_names(),
        )
        self._save_simulation_results(results=results_df, skipfirst=skipfirst)

    def simulate(
        self,
        t_end: float,
        steps: int | None = None,
    ) -> Self:
        """Simulate the model.

        Examples:
            >>> s.simulate(t_end=100)
            >>> s.simulate(t_end=100, steps=100)

        You can either supply only a terminal time point, or additionally also the
        number of steps for which values should be returned.

        Args:
            t_end: Terminal time point for the simulation.
            steps: Number of steps for the simulation.

        Returns:
            Self: The Simulator instance with updated results.

        """
        time, results = self.integrator.integrate(t_end=t_end, steps=steps)
        self._handle_simulation_results(time, results, skipfirst=True)
        return self

    def simulate_time_course(self, time_points: ArrayLike) -> Self:
        """Simulate the model over a given set of time points.

        Examples:
            >>> Simulator(model).simulate_time_course([1, 2, 3])

        You can either supply only a terminal time point, or additionally also the
        number of steps or exact time points for which values should be returned.

        Args:
            t_end: Terminal time point for the simulation.
            steps: Number of steps for the simulation.
            time_points: Exact time points for which values should be returned.

        Returns:
            Self: The Simulator instance with updated results.

        """
        time, results = self.integrator.integrate_time_course(time_points=time_points)
        self._handle_simulation_results(time, results, skipfirst=True)
        return self

    def simulate_to_steady_state(
        self,
        tolerance: float = 1e-6,
        *,
        rel_norm: bool = False,
    ) -> Self:
        """Simulate the model to steady state.

        Examples:
            >>> Simulator(model).simulate_to_steady_state()
            >>> Simulator(model).simulate_to_steady_state(tolerance=1e-8)
            >>> Simulator(model).simulate_to_steady_state(rel_norm=True)

        You can either supply only a terminal time point, or additionally also the
        number of steps or exact time points for which values should be returned.

        Args:
            tolerance: Tolerance for the steady-state calculation.
            rel_norm: Whether to use relative norm for the steady-state calculation.

        Returns:
            Self: The Simulator instance with updated results.

        """
        time, results = self.integrator.integrate_to_steady_state(
            tolerance=tolerance,
            rel_norm=rel_norm,
        )
        self._handle_simulation_results(
            [time] if time is not None else None,
            [results] if results is not None else None,  # type: ignore
            skipfirst=False,
        )
        return self

    def simulate_over_protocol(
        self,
        protocol: pd.DataFrame,
        time_points_per_step: int = 10,
    ) -> Self:
        """Simulate the model over a given protocol.

        Examples:
            >>> Simulator(model).simulate_over_protocol(
            ...     protocol,
            ...     time_points_per_step=10
            ... )

        Args:
            protocol: DataFrame containing the protocol.
            time_points_per_step: Number of time points per step.

        Returns:
            The Simulator instance with updated results.

        """
        for t_end, pars in protocol.iterrows():
            t_end = cast(pd.Timedelta, t_end)
            self.model.update_parameters(pars.to_dict())
            self.simulate(t_end.total_seconds(), steps=time_points_per_step)
            if self.concs is None:
                break
        return self

    def _get_dependent_vectorised(
        self,
        concs: list[pd.DataFrame],
        params: list[dict[str, float]],
        *,
        include_readouts: bool = True,
    ) -> list[pd.DataFrame]:
        dependent: list[pd.DataFrame] = []

        for res, p in zip(concs, params, strict=True):
            self.model.update_parameters(p)
            dependent.append(
                self.model.get_dependent_time_course(
                    concs=res,
                    include_readouts=include_readouts,
                )
            )
        return dependent

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
        """Get the concentration results.

        Examples:
            >>> Simulator(model).get_concs()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        Returns:
            pd.DataFrame: DataFrame of concentrations.

        """
        if self.concs is None:
            return None

        results = self.concs.copy()
        if normalise is not None:
            results = _normalise_split_results(results=results, normalise=normalise)
        if concatenated:
            return pd.concat(results, axis=0)

        return results

    # Get results depending on the raw results

    def _get_dependent(self) -> list[pd.DataFrame] | None:
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None

        return self._get_dependent_vectorised(concs, params)

    @overload
    def get_full_concs(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
        include_readouts: bool = True,
        dependent: list[pd.DataFrame] | None = None,
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
        include_readouts: bool = True,
        dependent: list[pd.DataFrame] | None = None,
    ) -> pd.DataFrame | None: ...

    @overload
    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
        dependent: list[pd.DataFrame] | None = None,
    ) -> pd.DataFrame | None: ...

    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
        dependent: list[pd.DataFrame] | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get the full concentration results, including derived quantities.

        Examples:
            >>> Simulator(model).get_full_concs()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        Returns: DataFrame of full concentrations.

        """
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None
        if dependent is None:
            dependent = self._get_dependent_vectorised(concs, params)

        # Filter to full concs
        names = (
            self.model.get_variable_names() + self.model.get_derived_variable_names()
        )
        if include_readouts:
            names.extend(self.model.get_readout_names())
        dependent = [i.loc[:, names] for i in dependent]

        if normalise is not None:
            dependent = _normalise_split_results(
                results=dependent,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(dependent, axis=0)
        return dependent

    @overload
    def get_fluxes(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
        dependent: list[pd.DataFrame] | None = None,
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
        dependent: list[pd.DataFrame] | None = None,
    ) -> pd.DataFrame | None: ...

    @overload
    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        dependent: list[pd.DataFrame] | None = None,
    ) -> pd.DataFrame | None: ...

    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        dependent: list[pd.DataFrame] | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get the flux results.

        Examples:
            >>> Simulator(model).get_fluxes()
            Time             v1         v2
            0.000000   1.000000   10.00000
            0.000100   0.999900   9.999000
            0.000200   0.999800   9.998000

        Returns:
            pd.DataFrame: DataFrame of fluxes.

        """
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None
        if dependent is None:
            dependent = self._get_dependent_vectorised(concs, params)

        # Filter to fluxes
        names = self.model.get_reaction_names()
        dependent = [i.loc[:, names] for i in dependent]
        if normalise is not None:
            dependent = _normalise_split_results(
                results=dependent,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(dependent, axis=0)
        return dependent

    def get_concs_and_fluxes(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Get the concentrations and fluxes.

        Examples:
            >>> Simulator(model).get_concs_and_fluxes()
            (concs, fluxes)


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple of concentrations and fluxes.

        """
        if (concs := self.concs) is None:
            return None, None
        if (params := self.simulation_parameters) is None:
            return None, None

        dependent = self._get_dependent_vectorised(concs, params)

        return self.get_concs(), self.get_fluxes(dependent=dependent)

    def get_full_concs_and_fluxes(
        self,
        *,
        include_readouts: bool = True,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Get the full concentrations and fluxes.

        >>> Simulator(model).get_full_concs_and_fluxes()
        (full_concs, full_fluxes)

        Args:
            include_readouts: Whether to include readouts in the results.

        Returns:
            Full concentrations and fluxes

        """
        if (concs := self.concs) is None:
            return None, None
        if (params := self.simulation_parameters) is None:
            return None, None
        dependent = self._get_dependent_vectorised(concs, params)

        return (
            self.get_full_concs(include_readouts=include_readouts, dependent=dependent),
            self.get_fluxes(dependent=dependent),
        )

    def get_results(self) -> pd.DataFrame | None:
        """Get the combined results of concentrations and fluxes.

        Examples:
            >>> Simulator(model).get_results()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   1.000000
            0.000100   0.999900   0.999900   0.999900   0.999900
            0.000200   0.999800   0.999800   0.999800   0.999800

        Returns:
            pd.DataFrame: Combined DataFrame of concentrations and fluxes.

        """
        c, v = self.get_concs_and_fluxes()
        if c is None or v is None:
            return None
        return pd.concat((c, v), axis=1)

    def get_full_results(self) -> pd.DataFrame | None:
        """Get the combined full results of concentrations and fluxes.

        Examples:
            >>> Simulator(model).get_full_results()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   1.000000
            0.000100   0.999900   0.999900   0.999900   0.999900
            0.000200   0.999800   0.999800   0.999800   0.999800

        """
        c, v = self.get_full_concs_and_fluxes()
        if c is None or v is None:
            return None
        return pd.concat((c, v), axis=1)

    def get_new_y0(self) -> dict[str, float] | None:
        """Get the new initial conditions after the simulation.

        Examples:
            >>> Simulator(model).get_new_y0()
            {"ATP": 1.0, "NADPH": 1.0}

        """
        if (res := self.get_concs()) is None:
            return None
        return dict(res.iloc[-1])

    def update_parameter(self, parameter: str, value: float) -> Self:
        """Updates the value of a specified parameter in the model.

        Examples:
            >>> Simulator(model).update_parameter("k1", 0.1)

        Args:
            parameter: The name of the parameter to update.
            value: The new value to set for the parameter.

        """
        self.model.update_parameter(parameter, value)
        return self

    def update_parameters(self, parameters: dict[str, float]) -> Self:
        """Updates the model parameters with the provided dictionary of parameters.

        Examples:
            >>> Simulator(model).update_parameters({"k1": 0.1, "k2": 0.2})

        Args:
            parameters: A dictionary where the keys are parameter names
                        and the values are the new parameter values.

        """
        self.model.update_parameters(parameters)
        return self

    def scale_parameter(self, parameter: str, factor: float) -> Self:
        """Scales the value of a specified parameter in the model.

        Examples:
            >>> Simulator(model).scale_parameter("k1", 0.1)

        Args:
            parameter: The name of the parameter to scale.
            factor: The factor by which to scale the parameter.

        """
        self.model.scale_parameter(parameter, factor)
        return self

    def scale_parameters(self, parameters: dict[str, float]) -> Self:
        """Scales the values of specified parameters in the model.

        Examples:
            >>> Simulator(model).scale_parameters({"k1": 0.1, "k2": 0.2})

        Args:
            parameters: A dictionary where the keys are parameter names
                        and the values are the scaling factors.

        """
        self.model.scale_parameters(parameters)
        return self
