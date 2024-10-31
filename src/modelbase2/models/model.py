"""The main class for modeling. Provides model construction and inspection tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd

from modelbase2.types import Array

if TYPE_CHECKING:
    from modelbase2.surrogates import AbstractSurrogate
    from modelbase2.types import Callable, DerivedFn, Iterable, Param, RetType, T

    # from . import LabelModel, LinearLabelModel


def invalidate_cache(method: Callable[Param, RetType]) -> Callable[Param, RetType]:
    def wrapper(
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        self = cast(Model, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


def _without(d: dict[str, T], key: str) -> dict[str, T]:
    new_d = d.copy()
    if key in d:
        new_d.pop(key)
    return new_d


def _sort_dependencies(
    available: set[str], elements: list[tuple[str, set[str]]], ctx: str
) -> list[str]:
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
            msg = (
                f"Exceeded max iterations on sorting {ctx}. "
                "Check if there are circular references.\n"
                f"Available: {unsorted}\n"
                f"Order: {order}"
            )
            raise ValueError(msg)
    return order


@dataclass(slots=True)
class Derived:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class DerivedVariable:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class DerivedParameter:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class DerivedStoichiometry:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class Reaction:
    fn: DerivedFn
    stoichiometry: dict[str, float | DerivedStoichiometry]
    args: list[str]


@dataclass(slots=True)
class Readout:
    fn: DerivedFn
    args: list[str]


@dataclass(slots=True)
class ModelCache:
    parameter_values: dict[str, float]
    stoich_by_cpds: dict[str, dict[str, float]]


@dataclass(
    slots=True,
)
class Model:
    _ids: dict[str, str] = field(default_factory=dict)
    _variables: dict[str, float] = field(default_factory=dict)
    _derived_variables: dict[str, DerivedVariable] = field(default_factory=dict)
    _parameters: dict[str, float] = field(default_factory=dict)
    _derived_parameters: dict[str, DerivedParameter] = field(default_factory=dict)
    _readouts: dict[str, Readout] = field(default_factory=dict)
    _reactions: dict[str, Reaction] = field(default_factory=dict)
    _surrogates: dict[str, AbstractSurrogate] = field(default_factory=dict)
    _cache: ModelCache | None = None

    ###########################################################################
    # Ids
    ###########################################################################

    def _insert_id(self, *, name: str, ctx: str) -> None:
        if name in self._ids:
            msg = f"Model already contains {ctx} called '{name}'"
            raise ValueError(msg)
        self._ids[name] = ctx

    def _remove_id(self, *, name: str) -> None:
        del self._ids[name]

    ##########################################################################
    # Parameters
    ##########################################################################

    @invalidate_cache
    def add_parameter(self, name: str, value: float) -> Self:
        self._insert_id(name=name, ctx="parameter")
        self._parameters[name] = value
        return self

    def add_parameters(self, parameters: dict[str, float]) -> Self:
        for k, v in parameters.items():
            self.add_parameter(k, v)
        return self

    def get_parameters(self) -> dict[str, float]:
        return self._parameters.copy()

    @invalidate_cache
    def remove_parameter(self, name: str) -> Self:
        self._remove_id(name=name)
        self._parameters.pop(name)
        return self

    def remove_parameters(self, names: list[str]) -> Self:
        for name in names:
            self.remove_parameter(name)
        return self

    @invalidate_cache
    def update_parameter(self, name: str, value: float) -> Self:
        if name not in self._parameters:
            msg = f"'{name}' not found in parameters"
            raise ValueError(msg)
        self._parameters[name] = value
        return self

    def update_parameters(self, parameters: dict[str, float]) -> Self:
        for k, v in parameters.items():
            self.update_parameter(k, v)
        return self

    def scale_parameter(self, name: str, factor: float) -> Self:
        return self.update_parameter(name, self._parameters[name] * factor)

    def scale_parameters(self, parameters: dict[str, float]) -> Self:
        for k, v in parameters.items():
            self.scale_parameter(k, v)
        return self

    ##########################################################################
    # Variables
    ##########################################################################

    @invalidate_cache
    def add_variable(
        self,
        name: str,
        initial_condition: float,
    ) -> Self:
        if name == "time":
            msg = "time is a protected variable for time"
            raise KeyError(msg)

        self._insert_id(name=name, ctx="variable")
        self._variables[name] = initial_condition
        return self

    def add_variables(
        self,
        variables: dict[str, float],
    ) -> Self:
        for name, y0 in variables.items():
            self.add_variable(name=name, initial_condition=y0)
        return self

    def get_variable_names(self) -> list[str]:
        return list(self._variables)

    @invalidate_cache
    def remove_variable(self, name: str) -> Self:
        self._remove_id(name=name)
        del self._variables[name]
        return self

    def remove_variables(self, variables: Iterable[str]) -> Self:
        for variable in variables:
            self.remove_variable(name=variable)
        return self

    def get_initial_conditions(self) -> list[float]:
        return list(self._variables.values())

    ##########################################################################
    # Derived
    ##########################################################################

    def _sort_derived_parameters(self) -> None:
        order = _sort_dependencies(
            available=set(self._parameters),
            elements=[(k, set(v.args)) for k, v in self._derived_parameters.items()],
            ctx="derived parameters",
        )
        # NOTE: this assumes dicts stay sorted by insertion order. Might break
        # in a future python release
        self._derived_parameters = {k: self._derived_parameters[k] for k in order}

    def _sort_derived_variables(self) -> None:
        order = _sort_dependencies(
            available=set(self._parameters) | set(self._variables) | {"time"},
            elements=[(k, set(v.args)) for k, v in self._derived_variables.items()],
            ctx="derived variables",
        )
        # NOTE: this assumes dicts stay sorted by insertion order. Might break
        # in a future python release
        self._derived_variables = {k: self._derived_variables[k] for k in order}

    @invalidate_cache
    def add_derived(
        self,
        name: str,
        fn: DerivedFn,
        args: list[str],
        *,
        sort_derived: bool = True,
    ) -> Self:
        if all(i in self._parameters for i in args):
            self._insert_id(name=name, ctx="derived parameter")
            self._derived_parameters[name] = DerivedParameter(fn, args)
            if sort_derived:
                self._sort_derived_parameters()
        else:
            self._insert_id(name=name, ctx="derived variable")
            self._derived_variables[name] = DerivedVariable(fn, args)
            if sort_derived:
                self._sort_derived_variables()
        return self

    def get_derived_parameter_names(self) -> list[str]:
        return list(self._derived_parameters)

    def get_derived_variable_names(self) -> list[str]:
        return list(self._derived_variables)

    @invalidate_cache
    def update_derived(
        self,
        name: str,
        fn: DerivedFn | None = None,
        args: list[str] | None = None,
        *,
        sort_derived: bool = True,
    ) -> Self:
        if (der := self._derived_parameters.get(name)) is not None:
            der.fn = der.fn if fn is None else fn
            der.args = der.args if args is None else args
            self._derived_parameters[name] = der
            if sort_derived:
                self._sort_derived_parameters()
        elif (dv := self._derived_variables.get(name)) is not None:
            dv.fn = dv.fn if fn is None else fn
            dv.args = dv.args if args is None else args
            self._derived_variables[name] = dv
            if sort_derived:
                self._sort_derived_variables()
        else:
            msg = f"Unknown derived parameter / variable: {name}"
            raise KeyError(msg)
        return self

    @invalidate_cache
    def remove_derived(self, name: str, *, sort_derived: bool = True) -> Self:
        self._remove_id(name=name)
        if name in self._derived_parameters:
            self._derived_parameters.pop(name)
            if sort_derived:
                self._sort_derived_parameters()
        else:
            self._derived_variables.pop(name)
            if sort_derived:
                self._sort_derived_variables()
        return self

    ###########################################################################
    # Reactions
    ###########################################################################

    @invalidate_cache
    def add_reaction(
        self,
        name: str,
        fn: DerivedFn,
        stoichiometry: dict[str, float | DerivedStoichiometry],
        args: list[str],
    ) -> Self:
        self._insert_id(name=name, ctx="reaction")
        self._reactions[name] = Reaction(fn=fn, stoichiometry=stoichiometry, args=args)
        return self

    def get_reaction_names(self) -> list[str]:
        return list(self._reactions)

    @invalidate_cache
    def update_reaction(
        self,
        name: str,
        fn: DerivedFn | None,
        stoichiometry: dict[str, float | DerivedStoichiometry] | None,
        args: list[str] | None,
    ) -> Self:
        rxn = self._reactions[name]
        rxn.fn = rxn.fn if fn is None else fn
        rxn.stoichiometry = (
            rxn.stoichiometry if stoichiometry is None else stoichiometry
        )
        rxn.args = rxn.args if args is None else args
        return self

    @invalidate_cache
    def remove_reaction(self, name: str) -> Self:
        self._remove_id(name=name)
        self._reactions.pop(name)
        return self

    ##########################################################################
    # Readouts
    # They are like derived variables, but only calculated on demand
    # Think of something like NADPH / (NADP + NADPH) as a proxy for energy state
    ##########################################################################

    def add_readout(self, name: str, function: DerivedFn, args: list[str]) -> Self:
        self._insert_id(name=name, ctx="readout")
        self._readouts[name] = Readout(function, args)
        return self

    def get_readout_names(self) -> list[str]:
        return list(self._readouts)

    def remove_readout(self, name: str) -> Self:
        self._remove_id(name=name)
        del self._readouts[name]
        return self

    ##########################################################################
    # Surrogates
    ##########################################################################

    @invalidate_cache
    def add_surrogate(
        self,
        name: str,
        surrogate: AbstractSurrogate,
    ) -> Self:
        self._insert_id(name=name, ctx="surrogate")
        self._surrogates[name] = surrogate
        return self

    ##########################################################################
    # Get args
    ##########################################################################

    def _create_cache(self) -> ModelCache:
        parameter_values = self._parameters.copy()
        for name, dp in self._derived_parameters.items():
            parameter_values[name] = dp.fn(*(parameter_values[i] for i in dp.args))

        stoich_by_compounds: dict[str, dict[str, float]] = {}
        for rxn_name, rxn in self._reactions.items():
            for cpd_name, factor in rxn.stoichiometry.items():
                d = stoich_by_compounds.setdefault(cpd_name, {})
                if isinstance(factor, DerivedStoichiometry):
                    d[rxn_name] = factor.fn(*(parameter_values[i] for i in factor.args))
                else:
                    d[rxn_name] = factor

        for surrogate in self._surrogates.values():
            for rxn_name, rxn in surrogate.stoichiometries.items():
                for cpd_name, factor in rxn.items():
                    stoich_by_compounds.setdefault(cpd_name, {})[rxn_name] = factor

        self._cache = ModelCache(
            parameter_values=parameter_values,
            stoich_by_cpds=stoich_by_compounds,
        )
        return self._cache

    def _get_args(
        self,
        concs: dict[str, float],
        time: float = 0.0,
        *,
        include_readouts: bool,
    ) -> dict[str, float]:
        if (cache := self._cache) is None:
            cache = self._create_cache()

        args = cache.parameter_values | concs
        args["time"] = time

        for name, dv in self._derived_variables.items():
            args[name] = dv.fn(*(args[arg] for arg in dv.args))

        if include_readouts:
            for name, ro in self._readouts.items():
                args[name] = ro.fn(*(args[arg] for arg in ro.args))
        return args

    def _get_args_vectorised(
        self,
        concs: pd.DataFrame,
        *,
        include_readouts: bool,
    ) -> pd.DataFrame:
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

        for name, dv in self._derived_variables.items():
            args[name] = dv.fn(*args.loc[:, dv.args].to_numpy().T)

        if include_readouts:
            for name, ro in self._readouts.items():
                args[name] = ro.fn(*args.loc[:, ro.args].to_numpy().T)
        return args

    def get_args(
        self,
        concs: dict[str, float],
        time: float = 0.0,
        *,
        include_readouts: bool = False,
    ) -> pd.Series[float]:
        return pd.Series(
            self._get_args(
                concs=concs,
                time=time,
                include_readouts=include_readouts,
            ),
            dtype=float,
        )

    ##########################################################################
    # Get full concs
    ##########################################################################

    def get_full_concs(
        self,
        concs: dict[str, float],
        time: float = 0.0,
        *,
        include_readouts: bool = True,
    ) -> pd.Series[float]:
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

    def _get_fluxes(
        self,
        args: dict[str, float],
    ) -> dict[str, float]:
        fluxes: dict[str, float] = {}
        for name, rxn in self._reactions.items():
            fluxes[name] = rxn.fn(*(args[arg] for arg in rxn.args))

        for surrogate in self._surrogates.values():
            fluxes |= surrogate.predict(
                np.array([args[arg] for arg in surrogate.inputs])
            )
        return fluxes

    def _get_fluxes_vectorised(
        self,
        args: pd.DataFrame,
    ) -> pd.DataFrame:
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
            outputs = pd.DataFrame(
                [surrogate.predict(y) for y in args.loc[:, surrogate.inputs].to_numpy()]
            )
            flux_df = pd.concat((flux_df, outputs), axis=1)
        return flux_df

    def get_fluxes(
        self,
        concs: dict[str, float],
        time: float = 0.0,
    ) -> pd.Series[float]:
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

    ##########################################################################
    # Get rhs
    ##########################################################################

    def _get_rhs(self, /, time: float, concs: Array) -> Array:
        """Simulation version. Swaps t and y!

        This can't get kw-only args, as the integrators call it with pos-only
        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        var_names = self.get_variable_names()
        concsd: dict[str, float] = dict(
            zip(
                var_names,
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

        dxdt = pd.Series(np.zeros(len(var_names), dtype=float), index=var_names)
        for k, stoc in cache.stoich_by_cpds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * fluxes[flux]
        return cast(Array, dxdt.to_numpy())

    def get_right_hand_side(
        self,
        concs: dict[str, float],
        time: float = 0.0,
    ) -> pd.Series[float]:
        if (cache := self._cache) is None:
            cache = self._create_cache()
        var_names = self.get_variable_names()
        fluxes = self._get_fluxes(
            self._get_args(
                concs=concs,
                time=time,
                include_readouts=False,
            )
        )
        dxdt = pd.Series(np.zeros(len(var_names), dtype=float), index=var_names)
        for k, stoc in cache.stoich_by_cpds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * fluxes[flux]
        return dxdt

    ##########################################################################
    # Junk
    ##########################################################################

    # def copy(self) -> Model:
    #     """Create a deepcopy of the model.

    #     Returns
    #     -------
    #     model
    #         Deepcopy of the model object

    #     """
    #     return copy.deepcopy(self)  # type: ignore

    # def _element_difference(
    #     self, other: Model, attribute: str
    # ) -> list[str] | dict[str, list[str]] | None:
    #     self_collection = getattr(self, attribute)
    #     other_collection = getattr(other, attribute)
    #     difference = sorted(set(other_collection).difference(self_collection))
    #     if not difference:
    #         return None
    #     if attribute == "compounds":
    #         return difference
    #     return {k: other_collection[k] for k in difference}

    # def _element_intersection(
    #     self, other: Model, attribute: str
    # ) -> list[str] | dict[str, list[str]] | None:
    #     self_collection = getattr(self, attribute)
    #     other_collection = getattr(other, attribute)
    #     intersection: list[str] = sorted(
    #         set(self_collection).intersection(other_collection)
    #     )
    #     if not intersection:
    #         return None
    #     if attribute == "compounds":
    #         return intersection
    #     return {k: other_collection[k] for k in intersection}

    # def __add__(self, other: Model) -> Model:
    #     return self.copy().__iadd__(other)

    # def __iadd__(self, other: Self) -> Self:
    #     self.add(compounds=self._element_difference(other, "compounds"))  # type: ignore

    #     for k, v in other.get_parameters().items():
    #         if k not in other.derived_parameters:
    #             self.add_and_update_parameter(k, v)

    #     for k, dpar in other.derived_parameters.items():
    #         if k in self.derived_parameters:
    #             self.update_derived_parameter(k, dpar["function"], dpar["parameters"])
    #         else:
    #             self.add_derived_parameter(
    #                 k, function=dpar["function"], parameters=dpar["parameters"]
    #             )

    #     for name in other._algebraic_module_order:
    #         module = other.algebraic_modules[name]
    #         if name in self.algebraic_modules:
    #             self.update_algebraic_module(name, **dict(module))
    #         else:
    #             self.add_algebraic_module(name, **dict(module))

    #     for rate_name, cpd_dict in other.derived_stoichiometries.items():
    #         for cpd_name, der_stoich in cpd_dict.items():
    #             self._add_derived_stoichiometry(
    #                 rate_name=rate_name,
    #                 cpd_name=cpd_name,
    #                 derived_stoichiometry=der_stoich,
    #             )

    #     for attribute in [
    #         "rates",
    #         "stoichiometries",
    #         "functions",
    #         "readouts",
    #     ]:
    #         self.add(**{attribute: self._element_difference(other, attribute)})  # type: ignore
    #         self.update(**{attribute: self._element_intersection(other, attribute)})  # type: ignore
    #     return self

    # def __sub__(self, other: Model) -> Model:
    #     m: Model = self.copy()
    #     for attribute in [
    #         "compounds",
    #         "parameters",
    #         "algebraic_modules",
    #         "rates",
    #         "stoichiometries",
    #         "functions",
    #         "derived_stoichiometries",
    #         "readouts",
    #     ]:
    #         m.remove(**{attribute: self._element_intersection(other, attribute)})  # type: ignore
    #     return m

    # def __isub__(self, other: Self) -> Self:
    #     return cast(Self, self.copy().__sub__(other))

    # def _collect_used_parameters(self) -> set[str]:
    #     used_parameters = set()
    #     for par in self.derived_parameters.values():
    #         used_parameters.update(par["parameters"])
    #     for module in self.algebraic_modules.values():
    #         used_parameters.update(module.parameters)
    #     for rate in self.rates.values():
    #         used_parameters.update(rate.parameters)
    #     for cpd_dict in self.derived_stoichiometries.values():
    #         for der_stoich in cpd_dict.values():
    #             used_parameters.update(der_stoich.args)
    #     return used_parameters

    # ##########################################################################
    # # Reactions
    # ##########################################################################

    # def _add_derived_stoichiometry(
    #     self, rate_name: str, cpd_name: str, derived_stoichiometry: DerivedStoichiometry
    # ) -> None:
    #     self.derived_stoichiometries.setdefault(rate_name, {})[cpd_name] = (
    #         derived_stoichiometry
    #     )
    #     self._update_derived_stoichiometries()

    # def _update_derived_stoichiometries(self) -> None:
    #     parameters = self.get_all_parameters()
    #     for rate_name, d in self.derived_stoichiometries.items():
    #         for cpd_name, stoichiometry in d.items():
    #             new_stoichiometry = stoichiometry.function(
    #                 *(parameters[i] for i in stoichiometry.args)
    #             )

    #             self.stoichiometries.setdefault(rate_name, {}).update(
    #                 {cpd_name: new_stoichiometry}
    #             )
    #             self.stoichiometries_by_compounds.setdefault(cpd_name, {}).update(
    #                 {rate_name: new_stoichiometry}
    #             )

    # def _remove_derived_stoichiometry(self, rate_name: str) -> None:
    #     self.derived_stoichiometries.pop(rate_name, None)

    # def _remove_derived_stoichiometries(self, rate_names: list[str]) -> None:
    #     for rate_name in rate_names:
    #         self._remove_derived_stoichiometry(rate_name)

    # def update_parameter(
    #     self,
    #     parameter_name: str,
    #     parameter_value: float,
    #     *,
    #     update_derived: bool = True,
    # ) -> Self:
    #     super().update_parameter(
    #         parameter_name,
    #         parameter_value,
    #         update_derived=update_derived,
    #     )
    #     self._update_derived_stoichiometries()
    #     return self

    # def update_parameters(
    #     self,
    #     parameters: dict[str, float],
    # ) -> Self:
    #     super().update_parameters(parameters)
    #     self._update_derived_stoichiometries()
    #     return self

    # def add_reaction(
    #     self,
    #     rate_name: str,
    #     function: DerivedFn,
    #     stoichiometry: dict[str, float],
    #     modifiers: list[str] | None = None,
    #     parameters: list[str] | None = None,
    #     dynamic_variables: list[str] | None = None,
    #     args: list[str] | None = None,
    #     *,
    #     reversible: bool = False,
    #     check_consistency: bool = True,
    #     derived_stoichiometry: dict[str, DerivedStoichiometry] | None = None,
    # ) -> Self:
    #     """Add a reaction to the model.

    #     Shortcut for add_rate and add stoichiometry functions.

    #     See Also
    #     --------
    #     add_rate
    #     add_stoichiometry

    #     Examples
    #     --------
    #     >>> add_reaction(
    #     >>>     rate_name="v1",
    #     >>>     function=mass_action,
    #     >>>     stoichiometry={"X": -1, "Y": 1},
    #     >>>     parameters=["k2"],
    #     >>> )

    #     >>> add_reaction(
    #     >>>     rate_name="v1",
    #     >>>     function=reversible_mass_action,
    #     >>>     stoichiometry={"X": -1, "Y": 1},
    #     >>>     parameters=["k1_fwd", "k1_bwd"],
    #     >>>     reversible=True,
    #     >>> )

    #     """
    #     self.add_stoichiometry(rate_name=rate_name, stoichiometry=stoichiometry)

    #     if derived_stoichiometry is not None:
    #         for cpd_name, der in derived_stoichiometry.items():
    #             self._add_derived_stoichiometry(rate_name, cpd_name, der)

    #     # Use now updated stoichiometry
    #     stoichiometry = self.stoichiometries[rate_name]
    #     substrates = [k for k, v in stoichiometry.items() if v < 0]
    #     products = [k for k, v in stoichiometry.items() if v > 0]

    #     self.add_rate(
    #         rate_name=rate_name,
    #         function=function,
    #         substrates=substrates,
    #         products=products,
    #         dynamic_variables=dynamic_variables,
    #         modifiers=modifiers,
    #         parameters=parameters,
    #         reversible=reversible,
    #         args=args,
    #         check_consistency=check_consistency,
    #     )
    #     return self

    # def add_reaction_from_args(
    #     self,
    #     rate_name: str,
    #     function: DerivedFn,
    #     stoichiometry: dict[str, float],
    #     args: list[str],
    #     reversible: bool | None = None,
    #     derived_stoichiometry: dict[str, DerivedStoichiometry] | None = None,
    #     *,
    #     check_consistency: bool = True,
    # ) -> Self:
    #     self.add_stoichiometry(rate_name=rate_name, stoichiometry=stoichiometry)

    #     if derived_stoichiometry is not None:
    #         for cpd_name, der in derived_stoichiometry.items():
    #             self._add_derived_stoichiometry(rate_name, cpd_name, der)

    #     # Use now updated stoichiometry
    #     stoichiometry = self.stoichiometries[rate_name]

    #     modifiers = []
    #     parameters = []
    #     dynamic_variables = []

    #     par_names = self.get_all_parameter_names()
    #     for i in args:
    #         if i in par_names:
    #             parameters.append(i)
    #         elif i not in stoichiometry:
    #             modifiers.append(i)
    #             dynamic_variables.append(i)
    #         else:
    #             dynamic_variables.append(i)
    #     substrates = [k for k, v in stoichiometry.items() if v < 0]
    #     products = [k for k, v in stoichiometry.items() if v > 0]

    #     if reversible is None:
    #         if any(i in dynamic_variables for i in products):
    #             reversible = True
    #         else:
    #             reversible = False

    #     self.add_rate(
    #         rate_name=rate_name,
    #         function=function,
    #         substrates=substrates,
    #         products=products,
    #         modifiers=modifiers,
    #         parameters=parameters,
    #         dynamic_variables=dynamic_variables,
    #         args=args,
    #         reversible=reversible,
    #         check_consistency=check_consistency,
    #     )
    #     return self

    # def update_reaction(
    #     self,
    #     rate_name: str,
    #     function: DerivedFn | None = None,
    #     stoichiometry: dict[str, float] | None = None,
    #     modifiers: list[str] | None = None,
    #     parameters: list[str] | None = None,
    #     dynamic_variables: list[str] | None = None,
    #     args: list[str] | None = None,
    #     reversible: bool | None = None,
    #     *,
    #     check_consistency: bool = True,
    #     derived_stoichiometry: dict[str, DerivedStoichiometry] | None = None,
    # ) -> Self:
    #     """Update an existing reaction.

    #     See Also
    #     --------
    #     add_reaction
    #     update_rate
    #     update_stoichiometry

    #     """
    #     if derived_stoichiometry is not None:
    #         for cpd_name, der in derived_stoichiometry.items():
    #             self._add_derived_stoichiometry(rate_name, cpd_name, der)

    #     if stoichiometry is not None:
    #         self.update_stoichiometry(rate_name=rate_name, stoichiometry=stoichiometry)

    #     if stoichiometry is not None:
    #         stoichiometry = self.stoichiometries[rate_name]
    #         substrates = [k for k, v in stoichiometry.items() if v < 0]
    #         products = [k for k, v in stoichiometry.items() if v > 0]
    #     else:
    #         substrates = None  # type: ignore
    #         products = None  # type: ignore
    #     self.update_rate(
    #         rate_name=rate_name,
    #         function=function,
    #         substrates=substrates,
    #         products=products,
    #         modifiers=modifiers,
    #         parameters=parameters,
    #         reversible=reversible,
    #         dynamic_variables=dynamic_variables,
    #         args=args,
    #         check_consistency=check_consistency,
    #     )
    #     return self

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

    # def update_reactions(
    #     self,
    #     reactions: dict,
    # ) -> Self:
    #     for rate_name, reaction in reactions.items():
    #         self.update_reaction(rate_name, **reaction)
    #     return self

    # def update_reaction_from_args(
    #     self,
    #     rate_name: str,
    #     function: DerivedFn | None = None,
    #     stoichiometry: dict[str, float] | None = None,
    #     args: list[str] | None = None,
    #     reversible: bool | None = None,
    #     *,
    #     check_consistency: bool = True,
    #     derived_stoichiometry: dict[str, DerivedStoichiometry] | None = None,
    # ) -> Self:
    #     if stoichiometry is not None:
    #         self.update_stoichiometry(rate_name=rate_name, stoichiometry=stoichiometry)

    #     if derived_stoichiometry is not None:
    #         for cpd_name, der in derived_stoichiometry.items():
    #             self._add_derived_stoichiometry(rate_name, cpd_name, der)

    #     # Now properly updated
    #     stoichiometry = self.stoichiometries[rate_name]

    #     if function is None:
    #         function = self.rates[rate_name].function

    #     if args is not None:
    #         modifiers = []
    #         parameters = []
    #         dynamic_variables = []

    #         par_names = self.get_all_parameter_names()
    #         for i in args:
    #             if i in par_names:
    #                 parameters.append(i)
    #             elif i not in stoichiometry:
    #                 modifiers.append(i)
    #                 dynamic_variables.append(i)
    #             else:
    #                 dynamic_variables.append(i)
    #         substrates = [k for k, v in stoichiometry.items() if v < 0]
    #         products = [k for k, v in stoichiometry.items() if v > 0]
    #         if reversible is None:
    #             if any(i in dynamic_variables for i in products):
    #                 reversible = True
    #             else:
    #                 reversible = False
    #     else:
    #         substrates = self.rates[rate_name].substrates
    #         products = self.rates[rate_name].products
    #         modifiers = self.rates[rate_name].modifiers
    #         parameters = self.rates[rate_name].parameters
    #         dynamic_variables = self.rates[rate_name].dynamic_variables
    #         args = self.rates[rate_name].args

    #     self.update_rate(
    #         rate_name=rate_name,
    #         function=function,
    #         substrates=substrates,
    #         products=products,
    #         modifiers=modifiers,
    #         parameters=parameters,
    #         reversible=reversible,
    #         dynamic_variables=dynamic_variables,
    #         args=args,
    #         check_consistency=check_consistency,
    #     )
    #     return self

    # ##########################################################################
    # # Conversion functions
    # ##########################################################################

    # def make_parameter_dynamic(self, name: str) -> Self:
    #     if name in self.derived_parameters:
    #         self.remove_derived_parameter(parameter_name=name)
    #     else:
    #         self.remove_parameter(parameter_name=name)
    #     self.add_compound(name)

    #     # Change all modifiers / parameters etc. accordingly
    #     for rate_name, rate in self.rates.items():
    #         if name in rate.args:
    #             self.update_reaction_from_args(rate_name=rate_name, args=rate.args)
    #     return self

    # def make_compound_static(self, name: str, parameter_value: float) -> Self:
    #     self.remove_compound(name)
    #     self.add_parameter(parameter_name=name, parameter_value=parameter_value)

    #     # Change all modifiers / parameters etc. accordingly
    #     for rate_name, rate in self.rates.items():
    #         if name in rate.args:
    #             self.update_reaction_from_args(
    #                 rate_name=rate_name,
    #                 args=rate.args,
    #                 stoichiometry=_without(self.stoichiometries[rate_name], name),
    #             )

    #     # There can be rates which change `name` without having it as an argument
    #     stoichs_by_cpd = copy.deepcopy(self.stoichiometries_by_compounds)

    #     for rate_name in stoichs_by_cpd.get(name, {}):
    #         self.update_reaction_from_args(
    #             rate_name=rate_name,
    #             stoichiometry=_without(self.stoichiometries[rate_name], name),
    #         )
    #     return self

    # ##########################################################################
    # # Simulation functions
    # ##########################################################################

    # def get_full_concentration_dict(
    #     self,
    #     y: dict[str, float] | dict[str, Array] | ArrayLike | Array,
    #     t: float | ArrayLike | Array = 0.0,
    #     *,
    #     include_readouts: bool = False,
    # ) -> dict[str, Array]:
    #     """Calculate the derived variables (at time(s) t).

    #     Examples
    #     --------
    #     >>> get_full_concentration_dict(y=[0, 0])
    #     >>> get_full_concentration_dict(y={"X": 0, "Y": 0})

    #     """
    #     if isinstance(t, int | float):
    #         t = [t]
    #     t = np.array(t)
    #     if isinstance(y, dict):
    #         y = {k: np.ones(len(t)) * v for k, v in y.items()}
    #     else:
    #         y = dict(
    #             zip(self.get_compounds(), (np.ones((len(t), 1)) * y).T, strict=True)
    #         )

    #     fcd = {
    #         k: np.ones(len(t)) * v
    #         for k, v in self._get_fcd(
    #             t=t,
    #             y=y,  # type: ignore
    #         ).items()
    #     }
    #     if include_readouts:
    #         args = self.parameters | y
    #         for name, readout in self.readouts.items():
    #             fcd[name] = np.ones(len(t)) * readout.function(
    #                 *(args[i] for i in readout.args)
    #             )
    #     return fcd  # type: ignore

    # def get_derived_variables(self, y: dict[str, float]) -> pd.Series[float]:
    #     s = (
    #         pd.DataFrame(self.get_full_concentration_dict(y))
    #         .iloc[0]
    #         .drop(["time", *self.compounds])
    #     )
    #     s.name = "derived_variables"
    #     return s

    ##########################################################################
    # Model conversion functions
    ##########################################################################

    # def to_labelmodel(
    #     self,
    #     labelcompounds: dict[str, int],
    #     labelmaps: dict[str, list[int]],
    # ) -> LabelModel:
    #     """Create a LabelModel from this model.

    #     Examples
    #     --------
    #     >>> m = Model()
    #     >>> m.add_reaction(
    #             rate_name="TPI",
    #             function=reversible_mass_action_1_1,
    #             stoichiometry={"GAP": -1, "DHAP": 1},
    #             parameters=["kf_TPI", "kr_TPI"],
    #             reversible=True,
    #         )
    #     >>> labelcompounds = {"GAP": 3, "DHAP": 3}
    #     >>> labelmaps = {"TPI": [2, 1, 0]}
    #     >>> m.to_labelmodel(labelcompounds=labelcompounds, labelmaps=labelmaps)

    #     """
    #     from modelbase2.ode import LabelModel

    #     lm = LabelModel()
    #     lm.add_parameters(self.get_parameters())
    #     for compound in self.get_compounds():
    #         if compound in labelcompounds:
    #             lm.add_label_compound(
    #                 compound=compound, num_labels=labelcompounds[compound]
    #             )
    #         else:
    #             lm.add_compound(compound=compound)

    #     for module_name, module in self.algebraic_modules.items():
    #         lm.add_algebraic_module(
    #             module_name=module_name,
    #             function=module["function"],
    #             compounds=module["compounds"],
    #             derived_compounds=module["derived_compounds"],
    #             modifiers=module["modifiers"],
    #             parameters=module["parameters"],
    #         )

    #     for rate_name, rate in self.rates.items():
    #         if rate_name not in labelmaps:
    #             lm.add_reaction(
    #                 rate_name=rate_name,
    #                 function=rate["function"],
    #                 stoichiometry=self.stoichiometries[rate_name],
    #                 modifiers=rate["modifiers"],
    #                 parameters=rate["parameters"],
    #                 reversible=rate["reversible"],
    #             )
    #         else:
    #             lm.add_labelmap_reaction(
    #                 rate_name=rate_name,
    #                 function=rate["function"],
    #                 stoichiometry=cast(dict[str, int], self.stoichiometries[rate_name]),
    #                 labelmap=labelmaps[rate_name],
    #                 modifiers=rate["modifiers"],
    #                 parameters=rate["parameters"],
    #                 reversible=rate["reversible"],
    #             )
    #     return lm

    # def to_linear_labelmodel(
    #     self,
    #     labelcompounds: dict[str, int],
    #     labelmaps: dict[str, list[int]],
    #     *,
    #     show_warnings: bool = True,
    # ) -> LinearLabelModel:
    #     """Create a LinearLabelModel from this model.

    #     Watch out that for a linear label model reversible reactions have to be split
    #     into a forward and backward part.

    #     Examples
    #     --------
    #     >>> m = Model()
    #     >>> m.add_reaction(
    #     >>>     rate_name="TPI_fwd",
    #     >>>     function=_mass_action_1_1,
    #     >>>     stoichiometry={"GAP": -1, "DHAP": 1},
    #     >>>     parameters=["kf_TPI"],
    #     >>> )
    #     >>> m.add_reaction(
    #     >>>     rate_name="TPI_bwd",
    #     >>>     function=mass_action_1_1,
    #     >>>     stoichiometry={"DHAP": -1, "GAP": 1},
    #     >>>     parameters=["kr_TPI"],
    #     >>> )
    #     >>> labelcompounds = {"GAP": 3, "DHAP": 3}
    #     >>> labelmaps = {"TPI_fwd": [2, 1, 0], 'TPI_bwd': [2, 1, 0]}
    #     >>> m.to_linear_labelmodel(labelcompounds=labelcompounds, labelmaps=labelmaps)

    #     """
    #     from modelbase2 import LinearLabelModel

    #     lm = LinearLabelModel(_warn=show_warnings)
    #     for compound in self.get_compounds():
    #         if compound in labelcompounds:
    #             lm.add_compound(compound=compound, num_labels=labelcompounds[compound])

    #     for rate_name, rate in self.rates.items():
    #         if rate_name in labelmaps:
    #             if rate["reversible"] and show_warnings:
    #                 warnings.warn(
    #                     f"Reaction {rate_name} is annotated as reversible. "
    #                     "Did you remember to split it into a forward and backward part?",
    #                     stacklevel=1,
    #                 )
    #             lm.add_reaction(
    #                 rate_name=rate_name,
    #                 stoichiometry={
    #                     k: v
    #                     for k, v in cast(
    #                         dict[str, int], self.stoichiometries[rate_name]
    #                     ).items()
    #                     if k in labelcompounds
    #                 },
    #                 labelmap=labelmaps[rate_name],
    #             )
    #         elif show_warnings:
    #             warnings.warn(
    #                 f"Skipping reaction {rate_name} as no labelmap is given",
    #                 stacklevel=1,
    #             )
    #     return lm
