# from __future__ import annotations

# __all__ = [
#     "_LinearLabelSimulate",
# ]

# import copy
# from typing import TYPE_CHECKING, Any, Self, cast

# from modelbase2.ode.models import LinearLabelModel as _LinearLabelModel
# from modelbase2.types import Array, ArrayLike, Axis
# from modelbase2.utils.plotting import plot, plot_grid

# from . import _BaseSimulator

# if TYPE_CHECKING:
#     from collections.abc import Iterable

#     from matplotlib.figure import Figure

#     from modelbase2.ode.integrators import AbstractIntegrator as _AbstractIntegrator


# class _LinearLabelSimulate(_BaseSimulator[_LinearLabelModel]):
#     """Simulator for LinearLabelModels."""

#     def __init__(
#         self,
#         model: _LinearLabelModel,
#         integrator: type[_AbstractIntegrator],
#         y0: ArrayLike | None = None,
#         time: list[Array] | None = None,
#         results: list[Array] | None = None,
#     ) -> None:
#         self.y0: ArrayLike | None  # For some reasons mypy has problems finding this
#         super().__init__(
#             model=model,
#             integrator=integrator,
#             y0=y0,
#             results=results,
#         )

#     def _test_run(self) -> None:
#         if self.y0 is None:
#             msg = "y0 must not be None"
#             raise ValueError(msg)
#         self.model.get_fluxes_dict(
#             y=self.y0,
#             v_ss=self.model._v_ss,
#             external_label=self.model._external_label,
#         )
#         self.model.get_right_hand_side(
#             y_labels=self.y0,
#             y_ss=self.model._y_ss,
#             v_ss=self.model._v_ss,
#             external_label=self.model._external_label,
#             t=0,
#         )

#     def copy(self) -> _LinearLabelSimulate:
#         """Return a deepcopy of this class."""
#         new = copy.deepcopy(self)
#         if new.results is not None:
#             new._initialise_integrator(y0=new.results[-1])
#         elif new.y0 is not None:
#             new.initialise(
#                 label_y0=new.y0,
#                 y_ss=new.model._y_ss,
#                 v_ss=new.model._v_ss,
#                 external_label=new.model._external_label,
#                 test_run=False,
#             )
#         return new

#     def initialise(
#         self,
#         label_y0: ArrayLike | dict[str, float],
#         y_ss: dict[str, float],
#         v_ss: dict[str, float],
#         external_label: float = 1.0,
#         test_run: bool = True,
#     ) -> Self:
#         self.model._y_ss = y_ss
#         self.model._v_ss = v_ss
#         self.model._external_label = external_label
#         if self.results is not None:
#             self.clear_results()
#         if isinstance(label_y0, dict):
#             self.y0 = [label_y0[compound] for compound in self.model.get_compounds()]
#         else:
#             self.y0 = list(label_y0)
#         self._initialise_integrator(y0=self.y0)

#         if test_run:
#             self._test_run()
#         return self

#     def get_label_position(self, compound: str, position: int) -> Array | None:
#         """Get relative concentration of a single isotopomer.

#         Examples
#         --------
#         >>> get_label_position(compound="GAP", position=2)

#         """
#         res = self.get_results_dict(concatenated=True)
#         if res is None:
#             return None
#         return res[self.model.isotopomers[compound][position]]

#     def get_label_distribution(self, compound: str) -> Array | None:
#         """Get relative concentrations of all compound isotopomers.

#         Examples
#         --------
#         >>> get_label_position(compound="GAP")

#         """
#         compounds = self.model.isotopomers[compound]
#         res = self.get_results(concatenated=True)
#         if res is None:
#             return None
#         return cast(Array, res.loc[:, compounds].values)

#     def _make_legend_labels(
#         self, prefix: str, compound: str, initial_index: int
#     ) -> list[str]:
#         return [
#             f"{prefix}{i}"
#             for i in range(
#                 initial_index, len(self.model.isotopomers[compound]) + initial_index
#             )
#         ]
