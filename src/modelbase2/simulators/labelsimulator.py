# from __future__ import annotations

# __all__ = [
#     "_LabelSimulate",
# ]

# import re
# from typing import TYPE_CHECKING, Any, cast

# import numpy as np
# import pandas as pd

# from modelbase2.ode.models import LabelModel
# from modelbase2.types import Array, ArrayLike, Axis
# from modelbase2.utils.plotting import plot, plot_grid

# from .abstract_simulator import _BaseRateSimulator

# if TYPE_CHECKING:
#     from collections.abc import Iterable

#     from matplotlib.figure import Figure

#     from modelbase2.ode.integrators import AbstractIntegrator


# class _LabelSimulate(_BaseRateSimulator[LabelModel]):
#     """Simulator for LabelModels."""

#     def __init__(
#         self,
#         model: LabelModel,
#         integrator: type[AbstractIntegrator],
#         y0: ArrayLike | None = None,
#         results: list[pd.DataFrame] | None = None,
#         parameters: list[dict[str, float]] | None = None,
#     ) -> None:
#         super().__init__(
#             model=model,
#             integrator=integrator,
#             y0=y0,
#             results=results,
#             parameters=parameters,
#         )

#     def get_total_concentration(self, compound: str) -> Array | None:
#         """Get the total concentration of all isotopomers of a compound."""
#         res = self.get_full_results_dict(concatenated=True, include_readouts=False)
#         if res is None:
#             return None
#         return res[compound + "__total"]

#     def get_unlabeled_concentration(self, compound: str) -> Array | None:
#         """Get the concentration of an isotopomer that is unlabeled."""
#         carbons = "0" * model.label_compounds[compound]["num_labels"]
#         res = self.get_full_results_dict(include_readouts=False)
#         if res is None:
#             return None
#         return cast(dict[str, Array], res)[compound + f"__{carbons}"]

#     def get_total_label_concentration(self, compound: str) -> Array | None:
#         """Get the total concentration of all labeled isotopomers of a compound."""
#         total = self.get_total_concentration(compound=compound)
#         unlabeled = self.get_unlabeled_concentration(compound=compound)
#         if total is None or unlabeled is None:
#             return None
#         return cast(Array, total - unlabeled)

#     def get_all_isotopomer_concentrations_array(self, compound: str) -> Array | None:
#         """Get concentrations of all isotopomers of a compound."""
#         res = self.get_all_isotopomer_concentrations_df(compound=compound)
#         if res is None:
#             return None
#         return cast(Array, res.values)

#     def get_all_isotopomer_concentrations_dict(
#         self, compound: str
#     ) -> dict[str, Array] | None:
#         """Get concentrations of all isotopomers of a compound."""
#         res = self.get_all_isotopomer_concentrations_df(compound=compound)
#         if res is None:
#             return None
#         return dict(
#             zip(
#                 res.columns,
#                 res.values.T,
#                 strict=False,  # type: ignore
#             )
#         )

#     def get_all_isotopomer_concentrations_df(self, compound: str) -> pd.Series | None:
#         """Get concentrations of all isotopomers of a compound."""
#         isotopomers = model.get_compound_isotopomers(compound=compound)
#         df = self.get_results()
#         if isotopomers is None or df is None:
#             return None
#         df = cast(pd.DataFrame, df)[isotopomers]
#         return cast(pd.Series, df[isotopomers])

#     def get_concentrations_by_reg_exp_array(self, reg_exp: str) -> Array | None:
#         """Get concentrations of all isotopomers matching the regular expression."""
#         isotopomers = [i for i in model.get_compounds() if re.match(reg_exp, i)]
#         df = self.get_results()
#         if isotopomers is None or df is None:
#             return None
#         df = cast(pd.DataFrame, df)[isotopomers]
#         return cast(Array, df[isotopomers].values)

#     def get_concentrations_by_reg_exp_dict(
#         self, reg_exp: str
#     ) -> dict[str, Array] | None:
#         """Get concentrations of all isotopomers of a compound."""
#         isotopomers = [i for i in model.get_compounds() if re.match(reg_exp, i)]
#         df = self.get_results(concatenated=True)
#         if isotopomers is None or df is None:
#             return None
#         df = df[isotopomers]
#         return dict(zip(df.columns, df.values.T, strict=False))

#     def get_concentrations_by_reg_exp_df(self, reg_exp: str) -> pd.DataFrame | None:
#         """Get concentrations of all isotopomers of a compound."""
#         isotopomers = [i for i in model.get_compounds() if re.match(reg_exp, i)]
#         df = self.get_results(concatenated=True)
#         if isotopomers is None or df is None:
#             return None
#         df = df[isotopomers]
#         return df[isotopomers]

#     def get_concentration_at_positions(
#         self, compound: str, positions: int | list[int]
#     ) -> Array | None:
#         """Get concentration of an isotopomer labelled at certain position(s)."""
#         if isinstance(positions, int):
#             positions = [positions]
#         num_labels = model.label_compounds[compound]["num_labels"]
#         label_positions = ["[01]"] * num_labels
#         for position in positions:
#             label_positions[position] = "1"
#         reg_exp = f"{compound}__{''.join(label_positions)}"
#         res = self.get_concentrations_by_reg_exp_array(reg_exp=reg_exp)
#         if res is None:
#             return None
#         return cast(Array, np.sum(res, axis=1))

#     def get_concentrations_of_n_labeled_array(
#         self, compound: str, n_labels: int
#     ) -> Array | None:
#         """Get concentrations of all isotopomers that carry n labels."""
#         res = self.get_concentrations_of_n_labeled_df(
#             compound=compound, n_labels=n_labels
#         )
#         if res is None:
#             return None
#         return cast(Array, res.values)

#     def get_concentrations_of_n_labeled_dict(
#         self, compound: str, n_labels: int
#     ) -> dict[str, Array] | None:
#         """Get concentrations of all isotopomers that carry n labels."""
#         df = self.get_concentrations_of_n_labeled_df(
#             compound=compound, n_labels=n_labels
#         )
#         if df is None:
#             return None
#         return dict(zip(df.columns, df.values.T, strict=False))

#     def get_concentrations_of_n_labeled_df(
#         self, compound: str, n_labels: int
#     ) -> pd.DataFrame | None:
#         """Get concentrations of all isotopomers that carry n labels."""
#         isotopomers = model.get_compound_isotopomers_with_n_labels(
#             compound=compound,
#             n_labels=n_labels,
#         )
#         res = self.get_results(concatenated=True)
#         if res is None:
#             return None
#         return res[isotopomers]

#     def _make_legend_labels(
#         self, prefix: str, compound: str, initial_index: int
#     ) -> list[str]:
#         return [
#             f"{prefix}{i}"
#             for i in range(
#                 initial_index,
#                 model.get_compound_number_of_label_positions(compound) + initial_index,
#             )
#         ]
