from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LabelModel"]

import copy
import itertools as it
import warnings
from collections import defaultdict
from dataclasses import field
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pandas as pd

from modelbase2.types import (
    Array,
    ArrayLike,
    DerivedParameter,
    DerivedVariable,
    Reaction,
    Readout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from modelbase2.types import Callable, DerivedFn, Iterable, Param, RetType

    from . import Model


def total_concentration(
    *args: float,
) -> Array:
    """Return concentration of all isotopomers.

    Algebraic module function to keep track of the total
    concentration of a compound (so sum of its isotopomers).
    """
    return np.sum(args, axis=0)  # type: ignore


def _invalidate_cache(method: Callable[Param, RetType]) -> Callable[Param, RetType]:
    def wrapper(
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        self = cast(Model, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


def _generate_binary_labels(
    *,
    base_name: str,
    num_labels: int,
) -> list[str]:
    """Create binary label string.

    Returns
    -------
    isotopomers : list(str)
        Returns a list of all label isotopomers of the compound

    Examples
    --------
    >>> _generate_binary_labels(base_name='cpd', num_labels=0)
    ['cpd']

    >>> _generate_binary_labels(base_name='cpd', num_labels=1)
    ['cpd__0', 'cpd__1']

    >>> _generate_binary_labels(base_name='cpd', num_labels=2)
    ['cpd__00', 'cpd__01', 'cpd__10', 'cpd__11']

    """
    if num_labels > 0:
        return [
            base_name + "__" + "".join(i)
            for i in it.product(("0", "1"), repeat=num_labels)
        ]
    return [base_name]


@dataclass(slots=True)
class ModelCache:
    parameter_values: dict[str, float]
    stoich_by_cpds: dict[str, dict[str, float]]


@dataclass(slots=True)
class LabelModel:
    _ids: dict[str, str] = field(default_factory=dict)
    _variables: dict[str, float] = field(default_factory=dict)
    _label_variables: dict[str, ...] = field(default_factory=dict)
    _derived_variables: dict[str, DerivedVariable] = field(default_factory=dict)
    _parameters: dict[str, float] = field(default_factory=dict)
    _derived_parameters: dict[str, DerivedParameter] = field(default_factory=dict)
    _readouts: dict[str, Readout] = field(default_factory=dict)
    _reactions: dict[str, Reaction] = field(default_factory=dict)
    _cache: ModelCache | None = None
    # self.label_compounds: dict[str, dict[str, Any]] = {}
    # self.nonlabel_compounds: list[str] = []
    # self.base_reactions: dict[str, dict[str, Any]] = {}

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
    @_invalidate_cache
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

    @_invalidate_cache
    def add_variable(  # type: ignore
        self,
        compound: str,
        *,
        is_isotopomer: bool = False,
    ) -> None:
        """Add a single compound to the model.

        Parameters
        ----------
        compound
            Name / id of the compound
        is_isotopomer
            Whether the compound is an isotopomer of a base compound
            or a non-label compound

        """
        super().add_compound(compound=compound)
        if not is_isotopomer:
            self.nonlabel_compounds.append(compound)

    def _add_base_compound(
        self,
        *,
        base_compound: str,
        num_labels: int,
        label_names: list[str],
    ) -> None:
        """Add a base compound of label isotopomer."""
        self.label_compounds[base_compound] = {
            "num_labels": num_labels,
            "isotopomers": label_names,
        }
        self._check_and_insert_ids([base_compound], context="base_compound")

    def _add_isotopomers(
        self,
        *,
        base_compound: str,
        label_names: list[str],
    ) -> None:
        # Add all labelled compounds
        for compound in label_names:
            self.add_variable(compound=compound, is_isotopomer=True)

        # Create moiety for total compound concentration
        self.add_algebraic_module(
            module_name=base_compound + "__total",
            function=total_concentration,  # type: ignore
            compounds=label_names,
            derived_compounds=[base_compound + "__total"],
            modifiers=None,
            parameters=None,
        )

    def add_label_compound(
        self,
        compound: str,
        num_labels: int,
    ) -> None:
        """Create all label isotopomers and add them as compounds.

        Also create an algebraic module that tracks the total
        concentration of that compound

        Parameters
        ----------
        base_compound
            Base name of the compound
        num_labels
            Number of labels in the compound

        Warns
        -----
        UserWarning
            If compound is already in the model

        """
        if compound in self.label_compounds:
            warnings.warn(
                f"Overwriting compound {compound}",
                stacklevel=1,
            )
            self.remove_label_compound(compound=compound)
        if num_labels == 0:
            self.add_variable(
                compound=compound,
                is_isotopomer=False,
            )
        else:
            label_names = self._generate_binary_labels(
                base_name=compound, num_labels=num_labels
            )
            self._add_base_compound(
                base_compound=compound,
                num_labels=num_labels,
                label_names=label_names,
            )
            self._add_isotopomers(base_compound=compound, label_names=label_names)

    def add_label_compounds(
        self,
        compounds: dict[str, int],
    ) -> None:
        """Add multiple label-containing compounds to the model.

        Parameters
        ----------
        compounds
            {compound: num_labels}

        Examples
        --------
        >>> add_label_compounds({"GAP": 3, "DHAP": 3, "FBP": 6})

        See Also
        --------
        add_label_compound

        """

        for compound, num_labels in compounds.items():
            self.add_label_compound(compound=compound, num_labels=num_labels)

    def remove_compound(  # type: ignore
        self,
        compound: str,
        *,
        is_isotopomer: bool = False,
    ) -> None:
        """Remove a compound from the model.

        Parameters
        ----------
        compound
        is_isotopomer
            Whether the compound is an isotopomer of a base compound
            or a non-label compound

        """
        super().remove_compound(compound=compound)
        if not is_isotopomer:
            self.nonlabel_compounds.remove(compound)

    def remove_label_compound(
        self,
        compound: str,
    ) -> None:
        """Remove a label compound from the model."""
        label_compound = self.label_compounds.pop(compound)
        self._remove_ids([compound])
        for key in label_compound["isotopomers"]:
            self.remove_compound(compound=key, is_isotopomer=True)

    def remove_label_compounds(
        self,
        compounds: list[str],
    ) -> None:
        """Remove label compounds."""
        for compound in compounds:
            self.remove_label_compound(compound=compound)

    def get_base_compounds(
        self,
    ) -> list[str]:
        """Get all base compounds and non-label compounds."""
        return sorted(list(self.label_compounds) + self.nonlabel_compounds)

    def get_compound_number_of_label_positions(
        self,
        compound: str,
    ) -> int:
        """Get the number of possible labels positions of a compound."""
        return int(self.label_compounds[compound]["num_labels"])

    def get_compound_isotopomers(
        self,
        compound: str,
    ) -> list[str]:
        """Get all isotopomers of a compound."""
        return list(self.label_compounds[compound]["isotopomers"])

    def get_compound_isotopomers_with_n_labels(
        self, compound: str, n_labels: int
    ) -> list[str]:
        """Get all isotopomers of a compound, that have excactly n labels."""
        label_positions = self.label_compounds[compound]["num_labels"]
        label_patterns = [
            ["1" if i in positions else "0" for i in range(label_positions)]
            for positions in it.combinations(range(label_positions), n_labels)
        ]
        return [f"{compound}__{''.join(i)}" for i in label_patterns]

    def get_compound_isotopomer_with_label_position(
        self, base_compound: str, label_position: int | list[int]
    ) -> str:
        """Get compound isotopomer with a given label position.

        Examples
        --------
        >>> add_label_compounds({"x": 2})
        >>> get_compound_isotopomer_with_label_position(x, 0) => x__10
        >>> get_compound_isotopomer_with_label_position(x, [0]) => x__10
        >>> get_compound_isotopomer_with_label_position(x, [0, 1]) => x__11

        """
        if isinstance(label_position, int):
            label_position = [label_position]
        return f"{base_compound}__" + "".join(
            "1" if idx in label_position else "0"
            for idx in range(self.label_compounds[base_compound]["num_labels"])
        )

    @staticmethod
    def _split_label_string(
        label: str,
        *,
        labels_per_compound: list[int],
    ) -> list[str]:
        """Split label string according to labels given in label list.

        The labels in the label list correspond to the number of
        label positions in the compound.

        Examples
        --------
        >>> _split_label_string(label="01", labels_per_compound=[2])
        ["01"]

        >>> _split_label_string(label="01", labels_per_compound=[1, 1])
        ["0", "1"]

        >>> _split_label_string(label="0011", labels_per_compound=[4])
        ["0011"]

        >>> _split_label_string(label="0011", labels_per_compound=[3, 1])
        ["001", "1"]

        >>> _split_label_string(label="0011", labels_per_compound=[2, 2])
        ["00", "11"]

        >>> _split_label_string(label="0011", labels_per_compound=[1, 3])
        ["0", "011"]

        """
        split_labels = []
        cnt = 0
        for i in range(len(labels_per_compound)):
            split_labels.append(label[cnt : cnt + labels_per_compound[i]])
            cnt += labels_per_compound[i]
        return split_labels

    @staticmethod
    def _map_substrates_to_products(
        *,
        rate_suffix: str,
        labelmap: list[int],
    ) -> str:
        """Map the rate_suffix to products using the labelmap."""
        return "".join([rate_suffix[i] for i in labelmap])

    @staticmethod
    def _unpack_stoichiometries(
        *, stoichiometries: dict[str, int]
    ) -> tuple[list[str], list[str]]:
        """Split stoichiometries into substrates and products.

        Parameters
        ----------
        stoichiometries : dict(str: int)

        Returns
        -------
        substrates : list(str)
        products : list(str)

        """
        substrates = []
        products = []
        for k, v in stoichiometries.items():
            if v < 0:
                substrates.extend([k] * -v)
            else:
                products.extend([k] * v)
        return substrates, products

    def _get_labels_per_compound(
        self,
        *,
        compounds: list[str],
    ) -> list[int]:
        """Get labels per compound.

        This is used for _split_label string. Adds 0 for non-label compounds,
        to show that they get no label.
        """
        labels_per_compound = []
        for compound in compounds:
            try:
                labels_per_compound.append(self.label_compounds[compound]["num_labels"])
            except KeyError as e:
                if compound not in self.get_compounds():
                    msg = f"Compound {compound} neither a compound nor a label compound"
                    raise KeyError(msg) from e
                labels_per_compound.append(0)
        return labels_per_compound

    @staticmethod
    def _repack_stoichiometries(
        *, new_substrates: list[str], new_products: list[str]
    ) -> dict[str, float]:
        """Pack substrates and products into stoichiometric dict."""
        new_stoichiometries: defaultdict[str, int] = defaultdict(int)
        for arg in new_substrates:
            new_stoichiometries[arg] -= 1
        for arg in new_products:
            new_stoichiometries[arg] += 1
        return dict(new_stoichiometries)

    @staticmethod
    def _assign_compound_labels(
        *, base_compounds: list[str], label_suffixes: list[str]
    ) -> list[str]:
        """Assign the correct suffixes."""
        new_compounds = []
        for i, compound in enumerate(base_compounds):
            if label_suffixes[i] != "":
                new_compounds.append(compound + "__" + label_suffixes[i])
            else:
                new_compounds.append(compound)
        return new_compounds

    def add_algebraic_module(
        self,
        module_name: str,
        function: Callable[..., float] | Callable[..., Iterable[float]],
        compounds: list[str] | None = None,
        derived_compounds: list[str] | None = None,
        modifiers: list[str] | None = None,
        parameters: list[str] | None = None,
        dynamic_variables: list[str] | None = None,
        args: list[str] | None = None,
        *,
        check_consistency: bool = True,
        sort_modules: bool = True,
    ) -> Self:
        """Add an algebraic module to the model.

        CAUTION: The Python function of the module has to return an iterable.
        The Python function will get the function arguments in the following order:
        [**compounds, **modifiers, **parameters]

        CAUTION: In a LabelModel context compounds and modifiers will be mapped to
        __total if a label_compound without the isotopomer suffix is supplied.

        Parameters
        ----------
        module_name
            Name of the module
        function
            Python method of the algebraic module
        compounds
            Names of compounds used for module
        derived_compounds
            Names of compounds which are calculated by the module
        modifiers
            Names of compounds which act as modifiers on the module
        parameters
            Names of the parameters which are passed to the function
        meta_info
            Meta info of the algebraic module. Allowed keys are
            {common_name, notes, database_links}

        Warns
        -----
        UserWarning
            If algebraic module is already in the model.

        Examples
        --------
        >>> def rapid_equilibrium(substrate, k_eq)-> None:
        >>>    x = substrate / (1 + k_eq)
        >>>    y = substrate * k_eq / (1 + k_eq)
        >>>    return x, y

        >>> add_algebraic_module(
        >>>     module_name="fast_eq",
        >>>     function=rapid_equilibrium,
        >>>     compounds=["A"],
        >>>     derived_compounds=["X", "Y"],
        >>>     parameters=["K"],
        >>> )

        """
        if compounds is not None:
            compounds = [
                i + "__total" if i in self.label_compounds else i for i in compounds
            ]
        if modifiers is not None:
            modifiers = [
                i + "__total" if i in self.label_compounds else i for i in modifiers
            ]
        if dynamic_variables is not None:
            dynamic_variables = [
                i + "__total" if i in self.label_compounds else i
                for i in dynamic_variables
            ]
        if args is not None:
            args = [i + "__total" if i in self.label_compounds else i for i in args]
        super().add_algebraic_module(
            module_name=module_name,
            function=function,
            compounds=compounds,
            derived_compounds=derived_compounds,
            modifiers=modifiers,
            parameters=parameters,
            dynamic_variables=dynamic_variables,
            args=args,
            check_consistency=check_consistency,
            sort_modules=sort_modules,
        )
        return self

    def _get_external_labels(
        self,
        *,
        rate_name: str,
        total_product_labels: int,
        total_substrate_labels: int,
        external_labels: list[int] | None,
    ) -> str:
        n_external_labels = total_product_labels - total_substrate_labels
        if n_external_labels > 0:
            if external_labels is None:
                warnings.warn(
                    f"Added external labels for reaction {rate_name}",
                    stacklevel=1,
                )
                external_label_string = ["1"] * n_external_labels
            else:
                external_label_string = ["0"] * n_external_labels
                for label_pos in external_labels:
                    external_label_string[label_pos] = "1"
            return "".join(external_label_string)
        return ""

    def add_reaction(
        self,
        rate_name: str,
        function: Callable[..., float],
        stoichiometry: dict[str, float],
        modifiers: list[str] | None = None,
        parameters: list[str] | None = None,
        dynamic_variables: list[str] | None = None,
        args: list[str] | None = None,
        *,
        reversible: bool = False,
        check_consistency: bool = True,
    ) -> Self:
        """Add a reaction to the model.

        Shortcut for add_rate and add stoichiometry functions.

        See Also
        --------
        add_rate
        add_stoichiometry

        Examples
        --------
        >>> add_reaction(
        >>>     rate_name="v1",
        >>>     function=mass_action,
        >>>     stoichiometry={"X": -1, "Y": 1},
        >>>     parameters=["k2"],
        >>> )

        >>> add_reaction(
        >>>     rate_name="v1",
        >>>     function=reversible_mass_action,
        >>>     stoichiometry={"X": -1, "Y": 1},
        >>>     parameters=["k1_fwd", "k1_bwd"],
        >>>     reversible=True,
        >>> )

        """
        substrates = [k for k, v in stoichiometry.items() if v < 0]
        products = [k for k, v in stoichiometry.items() if v > 0]

        self.add_rate(
            rate_name=rate_name,
            function=function,
            substrates=substrates,
            products=products,
            dynamic_variables=dynamic_variables,
            modifiers=modifiers,
            parameters=parameters,
            reversible=reversible,
            args=args,
            check_consistency=check_consistency,
        )
        self.add_stoichiometry(rate_name=rate_name, stoichiometry=stoichiometry)
        return self

    def _add_base_reaction(
        self,
        *,
        rate_name: str,
        function: Callable,
        stoichiometry: dict[str, int],
        labelmap: list[int],
        external_labels: str | None,
        modifiers: list[str] | None,
        parameters: list[str] | None,
        reversible: bool,
        variants: list[str],
    ) -> None:
        self.base_reactions[rate_name] = {
            "function": function,
            "stoichiometry": stoichiometry,
            "labelmap": labelmap,
            "external_labels": external_labels,
            "modifiers": modifiers,
            "parameters": parameters,
            "reversible": reversible,
            "variants": variants,
        }

    def _create_isotopomer_reactions(
        self,
        *,
        rate_name: str,
        function: Callable,
        labelmap: list[int],
        modifiers: list[str] | None,
        parameters: list[str] | None,
        reversible: bool,
        external_labels: str,
        total_substrate_labels: int,
        base_substrates: list[str],
        base_products: list[str],
        labels_per_substrate: list[int],
        labels_per_product: list[int],
    ) -> list[str]:
        variants = []
        for rate_suffix in (
            "".join(i) for i in it.product(("0", "1"), repeat=total_substrate_labels)
        ):
            rate_suffix += external_labels  # noqa: PLW2901
            # This is the magic
            product_suffix = self._map_substrates_to_products(
                rate_suffix=rate_suffix, labelmap=labelmap
            )
            product_labels = self._split_label_string(
                label=product_suffix, labels_per_compound=labels_per_product
            )
            substrate_labels = self._split_label_string(
                label=rate_suffix, labels_per_compound=labels_per_substrate
            )

            new_substrates = self._assign_compound_labels(
                base_compounds=base_substrates, label_suffixes=substrate_labels
            )
            new_products = self._assign_compound_labels(
                base_compounds=base_products, label_suffixes=product_labels
            )
            new_stoichiometry = self._repack_stoichiometries(
                new_substrates=new_substrates, new_products=new_products
            )
            new_rate_name = rate_name + "__" + rate_suffix
            self.add_reaction(
                rate_name=new_rate_name,
                function=function,
                stoichiometry=new_stoichiometry,
                modifiers=modifiers,
                parameters=parameters,
                reversible=reversible,
                check_consistency=False,
            )
            variants.append(new_rate_name)
        return variants

    def add_labelmap_reaction(
        self,
        rate_name: str,
        function: Callable,
        stoichiometry: dict[str, int],
        labelmap: list[int],
        external_labels: list[int] | None = None,
        modifiers: list[str] | None = None,
        parameters: list[str] | None = None,
        *,
        reversible: bool = False,
    ) -> None:
        """Add a labelmap reaction.

        Parameters
        ----------
        rate_name
            Name of the rate function
        function
            Python method calculating the rate equation
        stoichiometry
            stoichiometry of the reaction
        labelmap
            Mapping of the product label positions to the substrates
        external_labels
            Positions in which external labels are supposed to be inserted
        modifiers
            Names of the modifiers. E.g time.
        parameters
            Names of the parameters
        reversible
            Whether the reaction is reversible.

        Examples
        --------
        >>> add_labelmap_reaction(
                rate_name="triose-phosphate-isomerase",
                function=reversible_mass_action,
                labelmap=[2, 1, 0],
                stoichiometry={"GAP": -1, "DHAP": 1},
                parameters=["kf_TPI", "kr_TPI"],
                reversible=True,
            )
        >>> add_labelmap_reaction(
                rate_name="aldolase",
                function=reversible_mass_action_two_one,
                labelmap=[0, 1, 2, 3, 4, 5],
                stoichiometry={"DHAP": -1, "GAP": -1, "FBP": 1},
                parameters=["kf_Ald", "kr_Ald"],
                reversible=True,
            )

        """
        if modifiers is not None:
            modifiers = [
                i + "__total" if i in self.label_compounds else i for i in modifiers
            ]

        base_substrates, base_products = self._unpack_stoichiometries(
            stoichiometries=stoichiometry
        )
        labels_per_substrate = self._get_labels_per_compound(compounds=base_substrates)
        labels_per_product = self._get_labels_per_compound(compounds=base_products)
        total_substrate_labels = sum(labels_per_substrate)
        total_product_labels = sum(labels_per_product)

        if len(labelmap) - total_substrate_labels < 0:
            msg = f"Labelmap 'missing' {abs(len(labelmap) - total_substrate_labels)} label(s)"
            raise ValueError(msg)

        external_label_str = self._get_external_labels(
            rate_name=rate_name,
            total_product_labels=total_product_labels,
            total_substrate_labels=total_substrate_labels,
            external_labels=external_labels,
        )

        variants = self._create_isotopomer_reactions(
            rate_name=rate_name,
            function=function,
            labelmap=labelmap,
            modifiers=modifiers,
            parameters=parameters,
            reversible=reversible,
            external_labels=external_label_str,
            total_substrate_labels=total_substrate_labels,
            base_substrates=base_substrates,
            base_products=base_products,
            labels_per_substrate=labels_per_substrate,
            labels_per_product=labels_per_product,
        )

        self._add_base_reaction(
            rate_name=rate_name,
            function=function,
            stoichiometry=stoichiometry,
            labelmap=labelmap,
            external_labels=external_label_str,
            modifiers=modifiers,
            parameters=parameters,
            reversible=reversible,
            variants=variants,
        )

    def update_labelmap_reaction(
        self,
        rate_name: str,
        function: Callable | None = None,
        stoichiometry: dict[str, int] | None = None,
        labelmap: list[int] | None = None,
        modifiers: list[str] | None = None,
        parameters: list[str] | None = None,
        reversible: bool | None = None,
    ) -> None:
        """Update an existing labelmap reaction.

        Parameters
        ----------
        rate_name
            Name of the rate function
        function
            Python method calculating the rate equation
        stoichiometry
            stoichiometry of the reaction
        labelmap
            Mapping of the product label positions to the substrates
        external_labels
            Positions in which external labels are supposed to be inserted
        modifiers
            Names of the modifiers. E.g time.
        parameters
            Names of the parameters
        reversible
            Whether the reaction is reversible.

        """
        if function is None:
            function = self.base_reactions[rate_name]["function"]
        if stoichiometry is None:
            stoichiometry = self.base_reactions[rate_name]["stoichiometry"]
        if labelmap is None:
            labelmap = self.base_reactions[rate_name]["labelmap"]
        if modifiers is None:
            modifiers = self.base_reactions[rate_name]["modifiers"]
        if parameters is None:
            parameters = self.base_reactions[rate_name]["parameters"]
        if reversible is None:
            reversible = self.base_reactions[rate_name]["reversible"]

        self.remove_labelmap_reaction(rate_name=rate_name)
        self.add_labelmap_reaction(
            rate_name=rate_name,
            function=function,  # type: ignore
            stoichiometry=stoichiometry,  # type: ignore
            labelmap=labelmap,  # type: ignore
            modifiers=modifiers,
            parameters=parameters,
            reversible=reversible,  # type: ignore
        )

    def remove_reaction(
        self,
        rate_name: str,
    ) -> None:
        """Remove a reaction from the model.

        Parameters
        ----------
        rate_name : str

        """
        self.remove_rate(rate_name=rate_name)
        self.remove_rate_stoichiometry(rate_name=rate_name)

    def remove_labelmap_reaction(
        self,
        rate_name: str,
    ) -> None:
        """Remove all variants of a base reaction.

        Parameters
        ----------
        rate_name : str
            Name of the rate

        """
        base_reaction = self.base_reactions.pop(rate_name)
        for rate in base_reaction["variants"]:
            if rate.startswith(rate_name):
                self.remove_reaction(rate_name=rate)

    def remove_labelmap_reactions(
        self,
        rate_names: list[str],
    ) -> None:
        """Remove all variants of a multiple labelmap reactions.

        Parameters
        ----------
        rate_names : iterable(str)

        See Also
        --------
        remove_labelmap_reaction

        """
        for rate_name in rate_names:
            self.remove_labelmap_reaction(rate_name=rate_name)

    def generate_y0(
        self,
        base_y0: ArrayLike | dict[str, float],
        label_positions: dict[str, int | list[int]] | None = None,
    ) -> dict[str, float]:
        """Generate y0 for all isotopomers given a base y0.

        Examples
        --------
        >>> base_y0 = {"GAP": 1, "DHAP": 0, "FBP": 0}
        >>> generate_y0(base_y0=base_y0, label_positions={"GAP": 0})
        >>> generate_y0(base_y0=base_y0, label_positions={"GAP": [0, 1, 2]})

        """
        if label_positions is None:
            label_positions = {}
        if not isinstance(base_y0, dict):
            base_y0 = dict(zip(self.label_compounds, base_y0, strict=True))

        y0 = dict(
            zip(self.get_compounds(), np.zeros(len(self.get_compounds())), strict=True)
        )
        for base_compound, concentration in base_y0.items():
            label_position = label_positions.get(base_compound, None)
            if label_position is None:
                try:
                    y0[self.label_compounds[base_compound]["isotopomers"][0]] = (
                        concentration
                    )
                except KeyError:  # non label compound
                    y0[base_compound] = concentration
            else:
                if isinstance(label_position, int):
                    label_position = [label_position]
                suffix = "__" + "".join(
                    "1" if idx in label_position else "0"
                    for idx in range(self.label_compounds[base_compound]["num_labels"])
                )
                y0[base_compound + suffix] = concentration
        return y0

    def get_full_concentration_dict(
        self,
        y: dict[str, float] | dict[str, Array] | ArrayLike | Array,
        t: float | ArrayLike | Array = 0.0,
        *,
        include_readouts: bool = False,
    ) -> dict[str, Array]:
        """Calculate the derived variables (at time(s) t).

        Examples
        --------
        >>> get_full_concentration_dict(y=[0, 0])
        >>> get_full_concentration_dict(y={"X": 0, "Y": 0})

        """
        if include_readouts:
            raise NotImplementedError
        if isinstance(t, int | float):
            t = [t]
        t = np.array(t)
        if isinstance(y, dict):
            y = {k: np.ones(len(t)) * v for k, v in y.items()}
        else:
            y = dict(
                zip(self.get_compounds(), (np.ones((len(t), 1)) * y).T, strict=True)
            )
        return {k: np.ones(len(t)) * v for k, v in self._get_fcd(t=t, y=y).items()}  # type: ignore

    def get_fluxes(
        self,
        y: (
            dict[str, float]
            | dict[str, ArrayLike]
            | dict[str, Array]
            | Array
            | ArrayLike
        ),
        t: float | ArrayLike | Array = 0.0,
    ) -> dict[str, Array]:
        """Calculate the fluxes at time point(s) t."""
        fcd = self.get_full_concentration_dict(y=y, t=t)  # type: ignore
        ones = np.ones(len(fcd["time"]))
        if len(fcd["time"]) == 1:
            return {k: ones * v for k, v in self._get_fluxes(fcd=fcd).items()}  # type: ignore
        return {k: ones * v for k, v in self._get_fluxes_from_df(fcd=fcd).items()}  # type: ignore

    def get_total_fluxes(
        self,
        rate_base_name: str,
        y: dict[str, float] | dict[str, ArrayLike],
        t: float | ArrayLike = 0,
    ) -> Array:
        """Get total fluxes of a base rate.

        Parameters
        ----------
        rate_base_name : str
        y : Union(iterable(num), dict(str: num))
        t : Union(num, iterable(num))

        Returns
        -------
        fluxes : numpy.array

        """
        rates = [i for i in self.rates if i.startswith(rate_base_name + "__")]
        return cast(
            Array,
            self.get_fluxes_df(y=y, t=t)[rates].sum(axis=1).values,  # type: ignore
        )

    def _create_label_scope_seed(
        self, *, initial_labels: dict[str, int] | dict[str, list[int]]
    ) -> dict[str, int]:
        """Create initial label scope seed."""
        # initialise all compounds with 0 (no label)
        labelled_compounds = {compound: 0 for compound in self.get_compounds()}

        # Set all unlabelled compounds to 1
        for name, compound in self.label_compounds.items():
            num_labels = compound["num_labels"]
            labelled_compounds[f"{name}__{'0' * num_labels}"] = 1
        # Also set all non-label compounds to 1
        for name in self.nonlabel_compounds:
            labelled_compounds[name] = 1
        # Set initial label
        for i in [
            self.get_compound_isotopomer_with_label_position(
                base_compound=base_compound, label_position=label_position
            )
            for base_compound, label_position in initial_labels.items()
        ]:
            labelled_compounds[i] = 1
        return labelled_compounds

    def get_label_scope(
        self,
        initial_labels: dict[str, int] | dict[str, list[int]],
    ) -> dict[int, set[str]]:
        """Return all label positions that can be reached step by step.

        Parameters
        ----------
        initial_labels : dict(str: num)

        Returns
        -------
        label_scope : dict{step : set of new positions}

        Examples
        --------
        >>> l.get_label_scope({"x": 0})
        >>> l.get_label_scope({"x": [0, 1], "y": 0})

        """
        labelled_compounds = self._create_label_scope_seed(
            initial_labels=initial_labels
        )
        new_labels = set("non empty entry to not fulfill while condition")
        # Loop until no new labels are inserted
        loop_count = 0
        result = {}
        while new_labels != set():
            new_cpds = labelled_compounds.copy()
            for rec, cpd_dict in self.get_stoichiometries().items():
                # Isolate substrates
                cpds = [i for i, j in cpd_dict.items() if j < 0]
                # Count how many of the substrates are 1
                i = 0
                for j in cpds:
                    i += labelled_compounds[j]
                # If all substrates are 1, set all products to 1
                if i == len(cpds):
                    for cpd in self.get_stoichiometries()[rec]:
                        new_cpds[cpd] = 1
                if self.rates[rec]["reversible"]:
                    # Isolate substrates
                    cpds = [i for i, j in cpd_dict.items() if j > 0]
                    # Count how many of the substrates are 1
                    i = 0
                    for j in cpds:
                        i += labelled_compounds[j]
                    # If all substrates are 1, set all products to 1
                    if i == len(cpds):
                        for cpd in self.get_stoichiometries()[rec]:
                            new_cpds[cpd] = 1
            # Isolate "old" labels
            s1 = pd.Series(labelled_compounds)
            s1 = cast(pd.Series, s1[s1 == 1])
            # Isolate new labels
            s2 = pd.Series(new_cpds)
            s2 = cast(pd.Series, s2[s2 == 1])
            # Find new labels
            new_labels = cast(set[str], set(s2.index).difference(set(s1.index)))
            # Break the loop once no new labels can be found
            if new_labels == set():
                break
            labelled_compounds = new_cpds
            result[loop_count] = new_labels
            loop_count += 1
        return result

    def _get_rhs(
        self,
        t: float | ArrayLike | Array,
        y: list[Array],
    ) -> Array:
        """Calculate the right hand side of the ODE system.

        This is the more performant version of get_right_hand_side()
        and thus returns only an array instead of a dictionary.

        Watch out that this function swaps t and y!
        """
        y = dict(zip(self.get_compounds(), np.array(y).reshape(-1, 1), strict=True))  # type: ignore
        fcd = self._get_fcd(t=t, y=y)  # type: ignore
        fluxes = self._get_fluxes(fcd=fcd)  # type: ignore
        compounds_local = self.get_compounds()
        dxdt = dict(zip(compounds_local, it.repeat(0.0)))
        for k, stoc in self.stoichiometries_by_compounds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * fluxes[flux]
        return np.array([dxdt[i] for i in compounds_local], dtype="float")
