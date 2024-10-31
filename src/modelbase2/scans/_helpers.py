import numpy as np
import pandas as pd
from attr import dataclass

from modelbase2.models import ModelProtocol
from modelbase2.types import Array


def _empty_conc_series(model: ModelProtocol) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.get_variable_names()), fill_value=np.nan),
        index=model.get_variable_names(),
    )


def _empty_flux_series(model: ModelProtocol) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.get_reaction_names()), fill_value=np.nan),
        index=model.get_reaction_names(),
    )


def _empty_conc_df(model: ModelProtocol, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_variable_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_variable_names(),
    )


def _empty_flux_df(model: ModelProtocol, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_reaction_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_reaction_names(),
    )


def empty_time_point(model: ModelProtocol) -> tuple[pd.Series, pd.Series]:
    return _empty_conc_series(model), _empty_flux_series(model)


def empty_time_course(
    model: ModelProtocol, time_points: Array
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _empty_conc_df(model, time_points), _empty_flux_df(model, time_points)


@dataclass(slots=True, init=False)
class TimePoint:
    concs: pd.Series
    fluxes: pd.Series

    def __init__(
        self,
        model: ModelProtocol,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
        idx: int = -1,
    ) -> None:
        self.concs = _empty_conc_series(model) if concs is None else concs.iloc[idx]
        self.fluxes = _empty_flux_series(model) if fluxes is None else fluxes.iloc[idx]


@dataclass(slots=True, init=False)
class TimeCourse:
    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def __init__(
        self,
        model: ModelProtocol,
        time_points: Array,
        concs: pd.DataFrame | None,
        fluxes: pd.DataFrame | None,
    ) -> None:
        self.concs = _empty_conc_df(model, time_points) if concs is None else concs
        self.fluxes = _empty_flux_df(model, time_points) if fluxes is None else fluxes
