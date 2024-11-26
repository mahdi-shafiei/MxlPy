"""The main class for modeling. Provides model construction and inspection tools."""

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
    from collections.abc import Mapping

    from modelbase2.surrogates import AbstractSurrogate
    from modelbase2.types import Callable, DerivedFn, Iterable, Param, RetType

    # from . import LabelModel, LinearLabelModel


class SortError(Exception):
    pass


def _invalidate_cache(method: Callable[Param, RetType]) -> Callable[Param, RetType]:
    def wrapper(
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        self = cast(Model, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


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
            raise SortError(msg)
    return order


def _select_derived_type(
    model: Model, el: Derived
) -> DerivedParameter | DerivedVariable:
    all_pars = set(model.get_parameter_names()) ^ set(
        model.get_derived_parameter_names()
    )
    if set(el.args).issubset(all_pars):
        return DerivedParameter(fn=el.fn, args=el.args)
    return DerivedVariable(fn=el.fn, args=el.args)


@dataclass(slots=True)
class ModelCache:
    var_names: list[str]
    parameter_values: dict[str, float]
    stoich_by_cpds: dict[str, dict[str, float]]
    dyn_stoich_by_cpds: dict[str, dict[str, DerivedVariable]]
    dxdt: pd.Series[float]


@dataclass(slots=True)
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
    # Cache
    ###########################################################################

    def _create_cache(self) -> ModelCache:
        parameter_values = self._parameters.copy()
        for name, dp in self._derived_parameters.items():
            parameter_values[name] = dp.fn(*(parameter_values[i] for i in dp.args))

        stoich_by_compounds: dict[str, dict[str, float]] = {}
        dyn_stoich_by_compounds: dict[str, dict[str, DerivedVariable]] = {}
        for rxn_name, rxn in self._reactions.items():
            for cpd_name, factor in rxn.stoichiometry.items():
                d_static = stoich_by_compounds.setdefault(cpd_name, {})

                if isinstance(factor, Derived):
                    dt = _select_derived_type(self, factor)

                    if isinstance(dt, DerivedParameter):
                        d_static[rxn_name] = dt.fn(
                            *(parameter_values[i] for i in factor.args)
                        )
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
            dxdt=dxdt,
        )
        return self._cache

    ###########################################################################
    # Ids
    ###########################################################################

    def _insert_id(self, *, name: str, ctx: str) -> None:
        if name == "time":
            msg = "time is a protected variable for time"
            raise KeyError(msg)

        if name in self._ids:
            msg = f"Model already contains {ctx} called '{name}'"
            raise NameError(msg)
        self._ids[name] = ctx

    def _remove_id(self, *, name: str) -> None:
        del self._ids[name]

    ##########################################################################
    # Parameters
    ##########################################################################

    @_invalidate_cache
    def add_parameter(self, name: str, value: float) -> Self:
        self._insert_id(name=name, ctx="parameter")
        self._parameters[name] = value
        return self

    def add_parameters(self, parameters: dict[str, float]) -> Self:
        for k, v in parameters.items():
            self.add_parameter(k, v)
        return self

    @property
    def parameters(self) -> dict[str, float]:
        return self._create_cache().parameter_values.copy()

    def get_parameter_names(self) -> list[str]:
        return list(self._parameters)

    @_invalidate_cache
    def remove_parameter(self, name: str) -> Self:
        self._remove_id(name=name)
        self._parameters.pop(name)
        return self

    def remove_parameters(self, names: list[str]) -> Self:
        for name in names:
            self.remove_parameter(name)
        return self

    @_invalidate_cache
    def update_parameter(self, name: str, value: float) -> Self:
        if name not in self._parameters:
            msg = f"'{name}' not found in parameters"
            raise NameError(msg)
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

    ##########################################################################
    # Variables
    ##########################################################################

    @_invalidate_cache
    def add_variable(
        self,
        name: str,
        initial_condition: float,
    ) -> Self:
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

    @_invalidate_cache
    def remove_variable(self, name: str) -> Self:
        self._remove_id(name=name)
        del self._variables[name]
        return self

    def remove_variables(self, variables: Iterable[str]) -> Self:
        for variable in variables:
            self.remove_variable(name=variable)
        return self

    @_invalidate_cache
    def update_variable(self, name: str, initial_condition: float) -> Self:
        self._variables[name] = initial_condition
        return self

    def get_variable_names(self) -> list[str]:
        return list(self._variables)

    def get_initial_conditions(self) -> dict[str, float]:
        return self._variables

    # def make_variable_static(self, name: str, parameter_value: float) -> Self:
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

    ##########################################################################
    # Derived
    ##########################################################################

    @property
    def derived_variables(self) -> dict[str, DerivedVariable]:
        return copy.deepcopy(self._derived_variables)

    @property
    def derived_parameters(self) -> dict[str, DerivedParameter]:
        return copy.deepcopy(self._derived_parameters)

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

    @_invalidate_cache
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

    @_invalidate_cache
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

    @_invalidate_cache
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

    @property
    def reactions(self) -> dict[str, Reaction]:
        return copy.deepcopy(self._reactions)

    @_invalidate_cache
    def add_reaction(
        self,
        name: str,
        fn: DerivedFn,
        stoichiometry: Mapping[str, float | Derived],
        args: list[str],
    ) -> Self:
        self._insert_id(name=name, ctx="reaction")
        self._reactions[name] = Reaction(fn=fn, stoichiometry=stoichiometry, args=args)
        return self

    def get_reaction_names(self) -> list[str]:
        return list(self._reactions)

    @_invalidate_cache
    def update_reaction(
        self,
        name: str,
        fn: DerivedFn | None,
        stoichiometry: dict[str, float | Derived] | None,
        args: list[str] | None,
    ) -> Self:
        rxn = self._reactions[name]
        rxn.fn = rxn.fn if fn is None else fn
        rxn.stoichiometry = (
            rxn.stoichiometry if stoichiometry is None else stoichiometry
        )
        rxn.args = rxn.args if args is None else args
        return self

    @_invalidate_cache
    def remove_reaction(self, name: str) -> Self:
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

    @_invalidate_cache
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

    def get_args_time_series(
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

    def get_fluxes_time_series(
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

    ##########################################################################
    # Get rhs
    ##########################################################################

    def __call__(self, /, time: float, concs: Array) -> Array:
        """Simulation version of get_right_hand_side. Swaps t and y!

        This can't get kw-only args, as the integrators call it with pos-only
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

    def get_right_hand_side(
        self,
        concs: dict[str, float],
        time: float = 0.0,
    ) -> pd.Series[float]:
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
