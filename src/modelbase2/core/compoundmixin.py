from __future__ import annotations

__all__ = [
    "CompoundMixin",
]

import warnings
from typing import TYPE_CHECKING, Self

from .basemodel import BaseModel
from .utils import convert_id_to_sbml, warning_on_one_line

if TYPE_CHECKING:
    from collections.abc import Iterable

    import libsbml

warnings.formatwarning = warning_on_one_line  # type: ignore


class CompoundMixin(BaseModel):
    """Mixin for compound functionality."""

    def __init__(self, compounds: list[str] | None = None) -> None:
        self.compounds: list[str] = []
        if compounds is not None:
            self.add_compounds(compounds=compounds)

    ##########################################################################
    # Compound functions
    ##########################################################################

    def add_compound(
        self,
        compound: str,
    ) -> Self:
        """Add a compound to the model.

        Parameters
        ----------
        compound
            Name / id of the compound

        """
        if not isinstance(compound, str):
            msg = "The compound name should be string"
            raise TypeError(msg)
        if compound == "time":
            msg = "time is a protected variable for time"
            raise KeyError(msg)
        if compound in self.compounds:
            warnings.warn(
                f"Overwriting compound {compound}",
                stacklevel=1,
            )
            self.remove_compound(compound=compound)
        self.compounds.append(compound)
        self._check_and_insert_ids([compound], context="add_compound")
        return self

    def add_compounds(
        self,
        compounds: list[str],
    ) -> Self:
        """Add multiple compounds to the model.

        See Also
        --------
        add_compound

        """
        for compound in compounds:
            self.add_compound(compound=compound)
        return self

    def remove_compound(self, compound: str) -> Self:
        """Remove a compound from the model"""
        self.compounds.remove(compound)
        self._remove_ids([compound])
        return self

    def remove_compounds(self, compounds: Iterable[str]) -> Self:
        """Remove compounds from the model"""
        for compound in compounds:
            self.remove_compound(compound=compound)
        return self

    def _get_all_compounds(self) -> list[str]:
        """Get all compounds from the model.

        If used together with the algebraic mixin, this will return
        compounds + derived_compounds. Here it's just for API stability
        """
        return list(self.compounds)

    def get_compounds(self) -> list[str]:
        """Get the compounds from the model"""
        return list(self.compounds)

    ##########################################################################
    # Source code functions
    ##########################################################################

    def _generate_compounds_source_code(self) -> str:
        """Generate modelbase source code for compounds.

        This is mainly used for the generate_model_source_code function.

        Returns
        -------
        compounds_modelbase_code : str
            Source code generating the modelbase compounds

        """
        if len(self.compounds) == 0:
            return ""
        return f"m.add_compounds(compounds={self.compounds!r})"

    ##########################################################################
    # SBML functions
    ##########################################################################

    def _create_sbml_compounds(self, *, sbml_model: libsbml.Model) -> None:
        """Create the compounds for the sbml model.

        Parameters
        ----------
        sbml_model : libsbml.Model

        """
        for compound_id in self.get_compounds():
            cpd = sbml_model.createSpecies()
            cpd.setId(convert_id_to_sbml(id_=compound_id, prefix="CPD"))

            cpd.setConstant(False)
            cpd.setBoundaryCondition(False)
            cpd.setHasOnlySubstanceUnits(False)
