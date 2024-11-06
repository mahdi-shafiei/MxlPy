from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelbase2.model import Model
from modelbase2.types import DerivedStoichiometry

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


def _generate_isotope_labels(base_name: str, num_labels: int) -> list[str]:
    """Returns a list of all label isotopomers of the compound."""
    if num_labels > 0:
        return [f"{base_name}__{i}" for i in range(num_labels)]
    msg = f"Compound {base_name} must have labels"
    raise ValueError(msg)


def _unpack_stoichiometries(
    stoichiometries: Mapping[str, float | DerivedStoichiometry],
) -> tuple[dict[str, int], dict[str, int]]:
    """Split stoichiometries into substrates and products."""
    substrates = {}
    products = {}
    for k, v in stoichiometries.items():
        if isinstance(v, DerivedStoichiometry):
            raise NotImplementedError

        if v < 0:
            substrates[k] = int(-v)
        else:
            products[k] = int(v)
    return substrates, products


def _stoichiometry_to_duplicate_list(stoichiometry: dict[str, int]) -> list[str]:
    long_form: list[str] = []
    for k, v in stoichiometry.items():
        long_form.extend([k] * v)
    return long_form


def _map_substrates_to_labelmap(
    substrates: list[str], labelmap: list[int]
) -> list[str]:
    return [substrates[i] for i in labelmap]


def _add_label_influx_or_efflux(
    substrates: list[str],
    products: list[str],
    labelmap: list[int],
) -> tuple[list[str], list[str]]:
    # Add label outfluxes
    if (diff := len(substrates) - len(products)) > 0:
        products.extend(["EXT"] * diff)

    # Label influxes
    if (diff := len(products) - len(substrates)) > 0:
        substrates.extend(["EXT"] * diff)

    # Broken labelmap
    if (diff := len(labelmap) - len(substrates)) < 0:
        msg = f"Labelmap 'missing' {abs(diff)} label(s)"
        raise ValueError(msg)
    return substrates, products


def relative_label_flux(label_percentage: float, v_ss: float) -> float:
    return label_percentage * v_ss


def one_div(y: float) -> float:
    return 1 / y


def neg_one_div(y: float) -> float:
    return -1 / y


@dataclass(slots=True)
class LinearLabelMapper:
    model: Model
    label_variables: dict[str, int] = field(default_factory=dict)
    label_maps: dict[str, list[int]] = field(default_factory=dict)

    def get_isotopomers(self, variables: list[str]) -> dict[str, list[str]]:
        isotopomers = {
            name: _generate_isotope_labels(name, num)
            for name, num in self.label_variables.items()
        }
        return {k: isotopomers[k] for k in variables}

    def build_model(
        self,
        concs: pd.Series,
        fluxes: pd.Series,
        external_label: float = 1.0,
        initial_labels: dict[str, int | list[int]] | None = None,
    ) -> Model:
        isotopomers = {
            name: _generate_isotope_labels(name, num)
            for name, num in self.label_variables.items()
        }
        variables = {k: 0.0 for iso in isotopomers.values() for k in iso}
        if initial_labels is not None:
            for base_compound, label_positions in initial_labels.items():
                if isinstance(label_positions, int):
                    label_positions = [label_positions]  # noqa: PLW2901
                for pos in label_positions:
                    variables[f"{base_compound}__{pos}"] = 1 / len(label_positions)

        m = Model()
        m.add_variables(variables)
        m.add_parameters(concs.to_dict() | fluxes.to_dict() | {"EXT": external_label})
        for rxn_name, label_map in self.label_maps.items():
            rxn = self.model._reactions[rxn_name]  # noqa: SLF001
            subs, prods = _unpack_stoichiometries(rxn.stoichiometry)

            subs = _stoichiometry_to_duplicate_list(subs)
            prods = _stoichiometry_to_duplicate_list(prods)
            subs = [j for i in subs for j in isotopomers[i]]
            prods = [j for i in prods for j in isotopomers[i]]
            subs, prods = _add_label_influx_or_efflux(subs, prods, label_map)
            subs = _map_substrates_to_labelmap(subs, label_map)
            for i, (substrate, product) in enumerate(zip(subs, prods, strict=True)):
                if substrate == product:
                    continue

                m.add_reaction(
                    name=f"{rxn_name}__{i}",
                    fn=relative_label_flux,
                    stoichiometry={
                        substrate: DerivedStoichiometry(
                            neg_one_div, [substrate.split("__")[0]]
                        ),
                        product: DerivedStoichiometry(
                            one_div, [product.split("__")[0]]
                        ),
                    },
                    args=[substrate, rxn_name],
                )
        return m
