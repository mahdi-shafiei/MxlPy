from typing import Self

import pandas as pd

from modelbase2.types import Array, Protocol


class ModelProtocol(Protocol):
    def get_variable_names(self) -> list[str]: ...
    def get_derived_variable_names(self) -> list[str]: ...
    def get_readout_names(self) -> list[str]: ...
    def get_reaction_names(self) -> list[str]: ...
    def get_parameters(self) -> dict[str, float]: ...
    def update_parameters(self, parameters: dict[str, float]) -> Self: ...

    # User-facing
    def get_args(
        self,
        concs: dict[str, float],
        time: float,
        *,
        include_readouts: bool = True,
    ) -> pd.Series: ...
    def get_full_concs(
        self,
        concs: dict[str, float],
        time: float,
        *,
        include_readouts: bool = True,
    ) -> pd.Series: ...
    def get_fluxes(
        self,
        concs: dict[str, float],
        time: float,
    ) -> pd.Series: ...
    def get_right_hand_side(
        self,
        concs: dict[str, float],
        time: float,
    ) -> pd.Series: ...

    # For integration
    def _get_rhs(self, /, time: float, concs: Array) -> Array: ...

    # Vectorised
    def _get_args_vectorised(
        self,
        concs: pd.DataFrame,
        *,
        include_readouts: bool,
    ) -> pd.DataFrame: ...
    def _get_fluxes_vectorised(
        self,
        args: pd.DataFrame,
    ) -> pd.DataFrame: ...
