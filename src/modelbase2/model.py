"""Model for Metabolic System Representation.

This module provides the core Model class and supporting functionality for representing
metabolic models, including reactions, variables, parameters and derived quantities.


The Model class supports:
    - Adding/removing reactions and variables
    - Parameter management
    - Derived quantity calculations
    - Model state caching
    - Import/export functionality
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd

from modelbase2.types import (
    Array,
    Derived,
    DerivedParameter,
    DerivedVariable,
    Reaction,
    Readout,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from modelbase2.surrogates import AbstractSurrogate
    from modelbase2.types import Callable, DerivedFn, Param, RetType


class SortError(Exception):
    """Raised when dependencies cannot be sorted topologically.

    This typically indicates circular dependencies in model components.
    """


def _invalidate_cache(method: Callable[Param, RetType]) -> Callable[Param, RetType]:
    """Decorator that invalidates model cache when decorated method is called.

    Args:
        method: Method to wrap with cache invalidation

    Returns:
        Wrapped method that clears cache before execution

    """

    def wrapper(
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        self = cast(Model, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


def _sort_dependencies(available: set[str], elements: list[tuple[str, set[str]]], ctx: str) -> list[str]:
    """Sort model elements topologically based on their dependencies.

    Args:
        available: Set of available component names
        elements: List of (name, dependencies) tuples to sort
        ctx: Context string for error messages

    Returns:
        List of element names in dependency order

    Raises:
        SortError: If circular dependencies are detected

    """
    from queue import Empty, SimpleQueue

    order = []
    max_iterations = len(elements)
    queue: SimpleQueue[tuple[str, set[str]]] = SimpleQueue()
    for k, v in elements:
        queue.put((k, v))

    last_name = None
    i = 0
    while True:
        try:
            new, args = queue.get_nowait()
        except Empty:
            break
        if args.issubset(available):
            available.add(new)
            order.append(new)
        else:
            if last_name == new:
                order.append(new)
                break
            queue.put((new, args))
            last_name = new
        i += 1

        # Failure case
        if i > max_iterations:
            unsorted = []
            while True:
                try:
                    unsorted.append(queue.get_nowait()[0])
                except Empty:
                    break
            msg = f"Exceeded max iterations on sorting {ctx}. " "Check if there are circular references.\n" f"Available: {unsorted}\n" f"Order: {order}"
            raise SortError(msg)
    return order


def _select_derived_type(model: Model, el: Derived) -> DerivedParameter | DerivedVariable:
    all_pars = set(model.get_parameter_names()) ^ set(model.get_derived_parameter_names())
    if set(el.args).issubset(all_pars):
        return DerivedParameter(fn=el.fn, args=el.args)
    return DerivedVariable(fn=el.fn, args=el.args)


@dataclass(slots=True)
class ModelCache:
    """ModelCache is a class that stores various model-related data structures.

    Attributes:
        var_names: A list of variable names.
        parameter_values: A dictionary mapping parameter names to their values.
        derived_parameters: A dictionary mapping parameter names to their derived parameter objects.
        derived_variables: A dictionary mapping variable names to their derived variable objects.
        stoich_by_cpds: A dictionary mapping compound names to their stoichiometric coefficients.
        dyn_stoich_by_cpds: A dictionary mapping compound names to their dynamic stoichiometric coefficients.
        dxdt: A pandas Series representing the rate of change of variables.

    """

    var_names: list[str]
    parameter_values: dict[str, float]
    derived_parameters: dict[str, DerivedParameter]
    derived_variables: dict[str, DerivedVariable]
    stoich_by_cpds: dict[str, dict[str, float]]
    dyn_stoich_by_cpds: dict[str, dict[str, DerivedVariable]]
    dxdt: pd.Series


@dataclass(slots=True)
class Model:
    """Represents a metabolic model.

    Attributes:
        _ids: Dictionary mapping internal IDs to names.
        _variables: Dictionary of model variables and their initial values.
        _parameters: Dictionary of model parameters and their values.
        _derived: Dictionary of derived quantities.
        _readouts: Dictionary of readout functions.
        _reactions: Dictionary of reactions in the model.
        _surrogates: Dictionary of surrogate models.
        _cache: Cache for storing model-related data structures.

    """

    _ids: dict[str, str] = field(default_factory=dict)
    _variables: dict[str, float] = field(default_factory=dict)
    _parameters: dict[str, float] = field(default_factory=dict)
    _derived: dict[str, Derived] = field(default_factory=dict)
    _readouts: dict[str, Readout] = field(default_factory=dict)
    _reactions: dict[str, Reaction] = field(default_factory=dict)
    _surrogates: dict[str, AbstractSurrogate] = field(default_factory=dict)
    _cache: ModelCache | None = None

    ###########################################################################
    # Cache
    ###########################################################################

    def _create_cache(self) -> ModelCache:
        """Creates and initializes the model cache.

        This method constructs a cache that includes parameter values, stoichiometry
        by compounds, dynamic stoichiometry by compounds, derived variables, and
        derived parameters. It processes the model's parameters, variables, derived
        elements, reactions, and surrogates to populate the cache.

        Returns:
            ModelCache: An instance of ModelCache containing the initialized cache data.

        """
        parameter_values: dict[str, float] = self._parameters.copy()
        all_parameter_names: set[str] = set(parameter_values)

        # Sort derived
        derived_order = _sort_dependencies(
            available=set(self._parameters) | set(self._variables) | {"time"},
            elements=[(k, set(v.args)) for k, v in self._derived.items()],
            ctx="derived",
        )

        # Split derived into parameters and variables
        derived_variables: dict[str, DerivedVariable] = {}
        derived_parameters: dict[str, DerivedParameter] = {}
        for name in derived_order:
            derived = self._derived[name]
            if all(i in all_parameter_names for i in derived.args):
                derived_parameters[name] = DerivedParameter(
                    fn=derived.fn,
                    args=derived.args,
                )
                all_parameter_names.add(name)
                parameter_values[name] = derived.fn(*(parameter_values[i] for i in derived.args))
            else:
                derived_variables[name] = DerivedVariable(
                    fn=derived.fn,
                    args=derived.args,
                )

        stoich_by_compounds: dict[str, dict[str, float]] = {}
        dyn_stoich_by_compounds: dict[str, dict[str, DerivedVariable]] = {}
        for rxn_name, rxn in self._reactions.items():
            for cpd_name, factor in rxn.stoichiometry.items():
                d_static = stoich_by_compounds.setdefault(cpd_name, {})

                if isinstance(factor, Derived):
                    dt = _select_derived_type(self, factor)

                    if isinstance(dt, DerivedParameter):
                        d_static[rxn_name] = dt.fn(*(parameter_values[i] for i in factor.args))
                    else:
                        dyn_stoich_by_compounds.setdefault(cpd_name, {})[rxn_name] = dt

                else:
                    d_static[rxn_name] = factor

        for surrogate in self._surrogates.values():
            for rxn_name, rxn in surrogate.stoichiometries.items():
                for cpd_name, factor in rxn.items():
                    stoich_by_compounds.setdefault(cpd_name, {})[rxn_name] = factor

        var_names = self.get_variable_names()
        dxdt = pd.Series(np.zeros(len(var_names), dtype=float), index=var_names)

        self._cache = ModelCache(
            var_names=var_names,
            parameter_values=parameter_values,
            stoich_by_cpds=stoich_by_compounds,
            dyn_stoich_by_cpds=dyn_stoich_by_compounds,
            derived_variables=derived_variables,
            derived_parameters=derived_parameters,
            dxdt=dxdt,
        )
        return self._cache

    ###########################################################################
    # Ids
    ###########################################################################

    @property
    def ids(self) -> dict[str, str]:
        """Returns a copy of the _ids dictionary.

        The _ids dictionary contains key-value pairs where both keys and values are strings.

        Returns:
            dict[str, str]: A copy of the _ids dictionary.

        """
        return self._ids.copy()

    def _insert_id(self, *, name: str, ctx: str) -> None:
        """Inserts an identifier into the model's internal ID dictionary.

        Args:
            name: The name of the identifier to insert.
            ctx: The context associated with the identifier.

        Raises:
            KeyError: If the name is "time", which is a protected variable.
            NameError: If the name already exists in the model's ID dictionary.

        """
        if name == "time":
            msg = "time is a protected variable for time"
            raise KeyError(msg)

        if name in self._ids:
            msg = f"Model already contains {ctx} called '{name}'"
            raise NameError(msg)
        self._ids[name] = ctx

    def _remove_id(self, *, name: str) -> None:
        """Remove an ID from the internal dictionary.

        Args:
            name (str): The name of the ID to be removed.

        Raises:
            KeyError: If the specified name does not exist in the dictionary.

        """
        del self._ids[name]

    ##########################################################################
    # Parameters
    ##########################################################################

    @_invalidate_cache
    def add_parameter(self, name: str, value: float) -> Self:
        """Adds a parameter to the model.

        Args:
            name (str): The name of the parameter.
            value (float): The value of the parameter.

        Returns:
            Self: The instance of the model with the added parameter.

        """
        self._insert_id(name=name, ctx="parameter")
        self._parameters[name] = value
        return self

    def add_parameters(self, parameters: dict[str, float]) -> Self:
        """Adds multiple parameters to the model.

        Args:
            parameters (dict[str, float]): A dictionary where the keys are parameter names
                                           and the values are the corresponding parameter values.

        Returns:
            Self: The instance of the model with the added parameters.

        """
        for k, v in parameters.items():
            self.add_parameter(k, v)
        return self

    @property
    def parameters(self) -> dict[str, float]:
        """Returns the parameters of the model.

        This method creates a cache of parameter values and returns a copy of it.

        Returns:
            parameters: A dictionary where the keys are parameter names (as strings)
                  and the values are parameter values (as floats).

        """
        return self._create_cache().parameter_values.copy()

    def get_parameter_names(self) -> list[str]:
        """Retrieve the names of the parameters.

        Returns:
            parametes: A list containing the names of the parameters.

        """
        return list(self._parameters)

    @_invalidate_cache
    def remove_parameter(self, name: str) -> Self:
        """Remove a parameter from the model.

        Args:
            name: The name of the parameter to remove.

        Returns:
            Self: The instance of the model with the parameter removed.

        """
        self._remove_id(name=name)
        self._parameters.pop(name)
        return self

    def remove_parameters(self, names: list[str]) -> Self:
        """Remove multiple parameters from the model.

        Args:
            names: A list of parameter names to be removed.

        Returns:
            Self: The instance of the model with the specified parameters removed.

        """
        for name in names:
            self.remove_parameter(name)
        return self

    @_invalidate_cache
    def update_parameter(self, name: str, value: float) -> Self:
        """Update the value of a parameter.

        Args:
            name: The name of the parameter to update.
            value: The new value for the parameter.

        Returns:
            Self: The instance of the class with the updated parameter.

        Raises:
            NameError: If the parameter name is not found in the parameters.

        """
        if name not in self._parameters:
            msg = f"'{name}' not found in parameters"
            raise NameError(msg)
        self._parameters[name] = value
        return self

    def update_parameters(self, parameters: dict[str, float]) -> Self:
        """Update multiple parameters of the model.

        Args:
            parameters: A dictionary where keys are parameter names and values are the new parameter values.

        Returns:
            Self: The instance of the model with updated parameters.

        """
        for k, v in parameters.items():
            self.update_parameter(k, v)
        return self

    def scale_parameter(self, name: str, factor: float) -> Self:
        """Scales the value of a specified parameter by a given factor.

        Args:
            name: The name of the parameter to be scaled.
            factor: The factor by which to scale the parameter's value.

        Returns:
            Self: The instance of the class with the updated parameter.

        """
        return self.update_parameter(name, self._parameters[name] * factor)

    def scale_parameters(self, parameters: dict[str, float]) -> Self:
        """Scales the parameters of the model.

        Args:
            parameters: A dictionary where the keys are parameter names
                        and the values are the scaling factors.

        Returns:
            Self: The instance of the model with scaled parameters.

        """
        for k, v in parameters.items():
            self.scale_parameter(k, v)
        return self

    @_invalidate_cache
    def make_parameter_dynamic(self, name: str, initial_value: float | None = None) -> Self:
        """Converts a parameter to a dynamic variable in the model.

        This method removes the specified parameter from the model and adds it as a variable with an optional initial value.

        Args:
            name: The name of the parameter to be converted.
            initial_value: The initial value for the new variable. If None, the current value of the parameter is used. Defaults to None.

        Returns:
            Self: The instance of the model with the parameter converted to a variable.

        """
        value = self._parameters[name] if initial_value is None else initial_value
        self.remove_parameter(name)
        self.add_variable(name, value)
        return self

    ##########################################################################
    # Variables
    ##########################################################################

    @property
    def variables(self) -> dict[str, float]:
        """Returns a copy of the variables dictionary.

        This method returns a copy of the internal dictionary that maps variable
        names to their corresponding float values.

        Returns:
            dict[str, float]: A copy of the variables dictionary.

        """
        return self._variables.copy()

    @_invalidate_cache
    def add_variable(self, name: str, initial_condition: float) -> Self:
        """Adds a variable to the model with the given name and initial condition.

        Args:
            name: The name of the variable to add.
            initial_condition: The initial condition value for the variable.

        Returns:
            Self: The instance of the model with the added variable.

        """
        self._insert_id(name=name, ctx="variable")
        self._variables[name] = initial_condition
        return self

    def add_variables(self, variables: dict[str, float]) -> Self:
        """Adds multiple variables to the model with their initial conditions.

        Args:
            variables: A dictionary where the keys are variable names (str)
                       and the values are their initial conditions (float).

        Returns:
            Self: The instance of the model with the added variables.

        """
        for name, y0 in variables.items():
            self.add_variable(name=name, initial_condition=y0)
        return self

    @_invalidate_cache
    def remove_variable(self, name: str) -> Self:
        """Remove a variable from the model.

        Args:
            name: The name of the variable to remove.

        Returns:
            Self: The instance of the model with the variable removed.

        """
        self._remove_id(name=name)
        del self._variables[name]
        return self

    def remove_variables(self, variables: Iterable[str]) -> Self:
        """Remove multiple variables from the model.

        Args:
            variables: An iterable of variable names to be removed.

        Returns:
            Self: The instance of the model with the specified variables removed.

        """
        for variable in variables:
            self.remove_variable(name=variable)
        return self

    @_invalidate_cache
    def update_variable(self, name: str, initial_condition: float) -> Self:
        """Updates the value of a variable in the model.

        Args:
            name: The name of the variable to update.
            initial_condition: The initial condition or value to set for the variable.

        Returns:
            Self: The instance of the model with the updated variable.

        """
        self._variables[name] = initial_condition
        return self

    def get_variable_names(self) -> list[str]:
        """Retrieve the names of all variables.

        Returns:
            variable_names: A list containing the names of all variables.

        """
        return list(self._variables)

    def get_initial_conditions(self) -> dict[str, float]:
        """Retrieve the initial conditions of the model.

        Returns:
            initial_conditions: A dictionary where the keys are variable names and the values are their initial conditions.

        """
        return self._variables

    def make_variable_static(self, name: str, value: float | None = None) -> Self:
        """Converts a variable to a static parameter.

        Args:
            name: The name of the variable to be made static.
            value: The value to assign to the parameter.
                   If None, the current value of the variable is used. Defaults to None.

        Returns:
            Self: The instance of the class for method chaining.

        """
        value = self._variables[name] if value is None else value
        self.remove_variable(name)
        self.add_parameter(name, value)
        return self

    ##########################################################################
    # Derived
    ##########################################################################

    @property
    def derived_variables(self) -> dict[str, DerivedVariable]:
        """Returns a dictionary of derived variables.

        Returns:
            derived_variables: A dictionary where the keys are strings
            representing the names of the derived variables and the values are
            instances of DerivedVariable.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        return cache.derived_variables

    @property
    def derived_parameters(self) -> dict[str, DerivedParameter]:
        """Returns a dictionary of derived parameters.

        Returns:
            dict[str, DerivedParameter]: A dictionary where the keys are
            parameter names and the values are DerivedParameter objects.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        return cache.derived_parameters

    @_invalidate_cache
    def add_derived(
        self,
        name: str,
        fn: DerivedFn,
        args: list[str],
    ) -> Self:
        """Adds a derived attribute to the model.

        Args:
            name: The name of the derived attribute.
            fn: The function used to compute the derived attribute.
            args: The list of arguments to be passed to the function.

        Returns:
            Self: The instance of the model with the added derived attribute.

        """
        self._derived[name] = Derived(fn, args)
        return self

    def get_derived_parameter_names(self) -> list[str]:
        """Retrieve the names of derived parameters.

        Returns:
            A list of names of the derived parameters.

        """
        return list(self.derived_parameters)

    def get_derived_variable_names(self) -> list[str]:
        """Retrieve the names of derived variables.

        Returns:
            A list of names of derived variables.

        """
        return list(self.derived_variables)

    @_invalidate_cache
    def update_derived(
        self,
        name: str,
        fn: DerivedFn | None = None,
        args: list[str] | None = None,
    ) -> Self:
        """Updates the derived function and its arguments for a given name.

        Args:
            name: The name of the derived function to update.
            fn: The new derived function. If None, the existing function is retained. Defaults to None.
            args: The new arguments for the derived function. If None, the existing arguments are retained. Defaults to None.

        Returns:
            Self: The instance of the class with the updated derived function and arguments.

        """
        der = self._derived[name]
        der.fn = der.fn if fn is None else fn
        der.args = der.args if args is None else args
        return self

    @_invalidate_cache
    def remove_derived(self, name: str) -> Self:
        """Remove a derived attribute from the model.

        Args:
            name: The name of the derived attribute to remove.

        Returns:
            Self: The instance of the model with the derived attribute removed.

        """
        self._remove_id(name=name)
        self._derived.pop(name)
        return self

    ###########################################################################
    # Reactions
    ###########################################################################

    @property
    def reactions(self) -> dict[str, Reaction]:
        """Retrieve the reactions in the model.

        Returns:
            dict[str, Reaction]: A deep copy of the reactions dictionary.

        """
        return copy.deepcopy(self._reactions)

    @_invalidate_cache
    def add_reaction(
        self,
        name: str,
        fn: DerivedFn,
        stoichiometry: Mapping[str, float | Derived],
        args: list[str],
    ) -> Self:
        """Adds a reaction to the model.

        Args:
            name: The name of the reaction.
            fn: The function representing the reaction.
            stoichiometry: The stoichiometry of the reaction, mapping species to their coefficients.
            args: A list of arguments for the reaction function.

        Returns:
            Self: The instance of the model with the added reaction.

        """
        self._insert_id(name=name, ctx="reaction")
        self._reactions[name] = Reaction(fn=fn, stoichiometry=stoichiometry, args=args)
        return self

    def get_reaction_names(self) -> list[str]:
        """Retrieve the names of all reactions.

        Returns:
            list[str]: A list containing the names of the reactions.

        """
        return list(self._reactions)

    @_invalidate_cache
    def update_reaction(
        self,
        name: str,
        fn: DerivedFn | None,
        stoichiometry: dict[str, float | Derived] | None,
        args: list[str] | None,
    ) -> Self:
        """Updates the properties of an existing reaction in the model.

        Args:
            name: The name of the reaction to update.
            fn: The new function for the reaction. If None, the existing function is retained.
            stoichiometry: The new stoichiometry for the reaction. If None, the existing stoichiometry is retained.
            args: The new arguments for the reaction. If None, the existing arguments are retained.

        Returns:
            Self: The instance of the model with the updated reaction.

        """
        rxn = self._reactions[name]
        rxn.fn = rxn.fn if fn is None else fn
        rxn.stoichiometry = rxn.stoichiometry if stoichiometry is None else stoichiometry
        rxn.args = rxn.args if args is None else args
        return self

    @_invalidate_cache
    def remove_reaction(self, name: str) -> Self:
        """Remove a reaction from the model by its name.

        Args:
            name: The name of the reaction to be removed.

        Returns:
            Self: The instance of the model with the reaction removed.

        """
        self._remove_id(name=name)
        self._reactions.pop(name)
        return self

    # def update_stoichiometry_of_cpd(
    #     self,
    #     rate_name: str,
    #     compound: str,
    #     value: float,
    # ) -> Model:
    #     self.update_stoichiometry(
    #         rate_name=rate_name,
    #         stoichiometry=self.stoichiometries[rate_name] | {compound: value},
    #     )
    #     return self

    # def scale_stoichiometry_of_cpd(
    #     self,
    #     rate_name: str,
    #     compound: str,
    #     scale: float,
    # ) -> Model:
    #     return self.update_stoichiometry_of_cpd(
    #         rate_name=rate_name,
    #         compound=compound,
    #         value=self.stoichiometries[rate_name][compound] * scale,
    #     )

    ##########################################################################
    # Readouts
    # They are like derived variables, but only calculated on demand
    # Think of something like NADPH / (NADP + NADPH) as a proxy for energy state
    ##########################################################################

    def add_readout(self, name: str, function: DerivedFn, args: list[str]) -> Self:
        """Adds a readout to the model.

        Args:
            name: The name of the readout.
            function: The function to be used for the readout.
            args: The list of arguments for the function.

        Returns:
            Self: The instance of the model with the added readout.

        """
        self._insert_id(name=name, ctx="readout")
        self._readouts[name] = Readout(function, args)
        return self

    def get_readout_names(self) -> list[str]:
        """Retrieve the names of all readouts.

        Returns:
            list[str]: A list containing the names of the readouts.

        """
        return list(self._readouts)

    def remove_readout(self, name: str) -> Self:
        """Remove a readout by its name.

        Args:
            name (str): The name of the readout to remove.

        Returns:
            Self: The instance of the class after the readout has been removed.

        """
        self._remove_id(name=name)
        del self._readouts[name]
        return self

    ##########################################################################
    # Surrogates
    ##########################################################################

    @_invalidate_cache
    def add_surrogate(
        self,
        name: str,
        surrogate: AbstractSurrogate,
    ) -> Self:
        """Adds a surrogate model to the current instance.

        Args:
            name (str): The name of the surrogate model.
            surrogate (AbstractSurrogate): The surrogate model instance to be added.

        Returns:
            Self: The current instance with the added surrogate model.

        """
        self._insert_id(name=name, ctx="surrogate")
        self._surrogates[name] = surrogate
        return self

    ##########################################################################
    # Get args
    ##########################################################################

    def _get_args(
        self,
        concs: dict[str, float],
        time: float = 0.0,
        *,
        include_readouts: bool,
    ) -> dict[str, float]:
        """Generate a dictionary of arguments for model calculations.

        Args:
            concs: A dictionary of concentrations with keys as the names of the substances
                   and values as their respective concentrations.
            time: The time point for the calculation
            include_readouts: A flag indicating whether to include readout values in the returned dictionary.

        Returns:
        -------
        dict[str, float]
            A dictionary containing parameter values, derived variables, and optionally readouts,
            with their respective names as keys and their calculated values as values.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()

        args = cache.parameter_values | concs
        args["time"] = time

        for name, dv in cache.derived_variables.items():
            args[name] = dv.fn(*(args[arg] for arg in dv.args))

        if include_readouts:
            for name, ro in self._readouts.items():
                args[name] = ro.fn(*(args[arg] for arg in ro.args))
        return args

    def get_args(
        self,
        concs: dict[str, float],
        time: float = 0.0,
        *,
        include_readouts: bool = False,
    ) -> pd.Series:
        """Generate a pandas Series of arguments for the model.

        Args:
            concs: A dictionary where keys are the names of the concentrations and values are their respective float values.
            time: The time point at which the arguments are generated (default is 0.0).
            include_readouts: Whether to include readouts in the arguments (default is False).

        Returns:
            A pandas Series containing the generated arguments with float dtype.

        """
        return pd.Series(
            self._get_args(
                concs=concs,
                time=time,
                include_readouts=include_readouts,
            ),
            dtype=float,
        )

    def get_args_time_series(
        self,
        concs: pd.DataFrame,
        *,
        include_readouts: bool,
    ) -> pd.DataFrame:
        """Generate a DataFrame containing time series arguments for model evaluation.

        Args:
            concs: A DataFrame containing concentration data with time as the index.
            include_readouts: If True, include readout variables in the resulting DataFrame.

        Returns:
            A DataFrame containing the combined concentration data, parameter values,
            derived variables, and optionally readout variables, with time as an additional column.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()

        pars_df = pd.DataFrame(
            np.full(
                (len(concs), len(cache.parameter_values)),
                np.fromiter(cache.parameter_values.values(), dtype=float),
            ),
            index=concs.index,
            columns=list(cache.parameter_values),
        )

        args = pd.concat((concs, pars_df), axis=1)
        args["time"] = args.index

        for name, dv in cache.derived_variables.items():
            args[name] = dv.fn(*args.loc[:, dv.args].to_numpy().T)

        if include_readouts:
            for name, ro in self._readouts.items():
                args[name] = ro.fn(*args.loc[:, ro.args].to_numpy().T)
        return args

    ##########################################################################
    # Get full concs
    ##########################################################################

    def get_full_concs(self, concs: dict[str, float], time: float = 0.0, *, include_readouts: bool = True) -> pd.Series:
        """Get the full concentrations as a pandas Series.

        Args:
            concs (dict[str, float]): A dictionary of concentrations with variable names as keys and their corresponding values as floats.
            time (float, optional): The time point at which to get the concentrations. Default is 0.0.
            include_readouts (bool, optional): Whether to include readout variables in the result. Default is True.

        Returns:
        pd.Series: A pandas Series containing the full concentrations for the specified variables.

        """
        names = self.get_variable_names() + self.get_derived_variable_names()
        if include_readouts:
            names.extend(self.get_readout_names())

        return self.get_args(
            concs=concs,
            time=time,
            include_readouts=include_readouts,
        ).loc[names]

    ##########################################################################
    # Get fluxes
    ##########################################################################

    def _get_fluxes(self, args: dict[str, float]) -> dict[str, float]:
        """Calculate the fluxes for the given arguments.

        Args:
            args (dict[str, float]): A dictionary where the keys are argument names and the values are their corresponding float values.

        Returns:
            dict[str, float]: A dictionary where the keys are reaction names and the values are the calculated fluxes.

        """
        fluxes: dict[str, float] = {}
        for name, rxn in self._reactions.items():
            fluxes[name] = rxn.fn(*(args[arg] for arg in rxn.args))

        for surrogate in self._surrogates.values():
            fluxes |= surrogate.predict(np.array([args[arg] for arg in surrogate.inputs]))
        return fluxes

    def get_fluxes(self, concs: dict[str, float], time: float = 0.0) -> pd.Series:
        """Calculate the fluxes for the given concentrations and time.

        Args:
            concs: A dictionary where keys are species names and values are their concentrations.
            time: The time at which to calculate the fluxes. Defaults to 0.0.

        Returns:
            Fluxes: A pandas Series containing the fluxes for each reaction.

        """
        args = self.get_args(
            concs=concs,
            time=time,
            include_readouts=False,
        )

        fluxes: dict[str, float] = {}
        for name, rxn in self._reactions.items():
            fluxes[name] = rxn.fn(*args.loc[rxn.args])

        for surrogate in self._surrogates.values():
            fluxes |= surrogate.predict(args.loc[surrogate.inputs].to_numpy())
        return pd.Series(fluxes, dtype=float)

    def get_fluxes_time_series(self, args: pd.DataFrame) -> pd.DataFrame:
        """Generate a time series of fluxes for the given reactions and surrogates.

        This method calculates the fluxes for each reaction in the model using the provided
        arguments and combines them with the outputs from the surrogates to create a complete
        time series of fluxes.

        Args:
            args (pd.DataFrame): A DataFrame containing the input arguments for the reactions
                                 and surrogates. Each column corresponds to a specific input
                                 variable, and each row represents a different time point.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated fluxes for each reaction and
                          the outputs from the surrogates. The index of the DataFrame matches
                          the index of the input arguments.

        """
        fluxes: dict[str, Array] = {}
        for name, rate in self._reactions.items():
            fluxes[name] = cast(
                Array,
                rate.fn(*args.loc[:, rate.args].to_numpy().T),
            )

        # Create df here already to avoid having to play around with
        # shape of surrogate outputs
        flux_df = pd.DataFrame(fluxes, index=args.index)
        for surrogate in self._surrogates.values():
            outputs = pd.DataFrame([surrogate.predict(y) for y in args.loc[:, surrogate.inputs].to_numpy()])
            flux_df = pd.concat((flux_df, outputs), axis=1)
        return flux_df

    ##########################################################################
    # Get rhs
    ##########################################################################

    def __call__(self, /, time: float, concs: Array) -> Array:
        """Simulation version of get_right_hand_side.

        Warning: Swaps t and y!
        This can't get kw-only args, as the integrators call it with pos-only

        Args:
            time: The current time point.
            concs: Array of concentrations


        Returns:
            The rate of change of each variable in the model.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        concsd: dict[str, float] = dict(
            zip(
                cache.var_names,
                concs,
                strict=True,
            )
        )
        args: dict[str, float] = self._get_args(
            concs=concsd,
            time=time,
            include_readouts=False,
        )
        fluxes: dict[str, float] = self._get_fluxes(args)

        dxdt = cache.dxdt
        dxdt[:] = 0
        for k, stoc in cache.stoich_by_cpds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * fluxes[flux]
        for k, sd in cache.dyn_stoich_by_cpds.items():
            for flux, dv in sd.items():
                n = dv.fn(*(args[i] for i in dv.args))
                dxdt[k] += n * fluxes[flux]
        return cast(Array, dxdt.to_numpy())

    def get_right_hand_side(self, concs: dict[str, float], time: float = 0.0) -> pd.Series:
        """Calculate the right-hand side of the differential equations for the model.

        Args:
            concs: A dictionary mapping compound names to their concentrations.
            time: The current time point. Defaults to 0.0.

        Returns:
            The rate of change of each variable in the model.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        var_names = self.get_variable_names()
        args = self._get_args(
            concs=concs,
            time=time,
            include_readouts=False,
        )
        fluxes = self._get_fluxes(args)
        dxdt = pd.Series(np.zeros(len(var_names), dtype=float), index=var_names)
        for k, stoc in cache.stoich_by_cpds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * fluxes[flux]

        for k, sd in cache.dyn_stoich_by_cpds.items():
            for flux, dv in sd.items():
                n = dv.fn(*(args[i] for i in dv.args))
                dxdt[k] += n * fluxes[flux]
        return dxdt
