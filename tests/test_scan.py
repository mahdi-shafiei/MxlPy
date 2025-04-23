from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelbase2 import Model, fns
from modelbase2.scan import (
    TimeCourse,
    TimePoint,
    _empty_conc_df,
    _empty_conc_series,
    _empty_flux_df,
    _empty_flux_series,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and,
    steady_state,
    time_course,
)


@pytest.fixture
def simple_model() -> Model:
    model = Model()
    model.add_parameters({"k1": 1.0, "k2": 2.0})
    model.add_variables({"S": 10.0, "P": 0.0})

    model.add_reaction(
        "v1",
        fn=fns.mass_action_1s,
        args=["S", "k1"],
        stoichiometry={"S": -1.0, "P": 1.0},
    )

    model.add_reaction(
        "v2",
        fn=fns.mass_action_1s,
        args=["P", "k2"],
        stoichiometry={"P": -1.0},
    )

    return model


def test_empty_conc_series(simple_model: Model) -> None:
    series = _empty_conc_series(simple_model)
    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert all(np.isnan(series))
    assert list(series.index) == ["S", "P"]


def test_empty_flux_series(simple_model: Model) -> None:
    series = _empty_flux_series(simple_model)
    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert all(np.isnan(series))
    assert list(series.index) == ["v1", "v2"]


def test_empty_conc_df(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    df = _empty_conc_df(simple_model, time_points)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert np.all(np.isnan(df.to_numpy()))
    assert np.all(df.index == time_points)
    assert np.all(df.columns == ["S", "P"])


def test_empty_flux_df(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    df = _empty_flux_df(simple_model, time_points)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert np.all(np.isnan(df.to_numpy()))
    assert np.all(df.index == time_points)
    assert np.all(df.columns == ["v1", "v2"])


def test_update_parameters_and(simple_model: Model) -> None:
    params = pd.Series({"k1": 2.0})

    def get_params(model: Model) -> dict[str, float]:
        return model.parameters

    result = _update_parameters_and(params, get_params, simple_model)
    assert result["k1"] == 2.0
    assert result["k2"] == 2.0  # Unchanged


def test_timepoint_from_scan(simple_model: Model) -> None:
    time_point = TimePoint.from_scan(simple_model, None, None)
    assert isinstance(time_point, TimePoint)
    assert isinstance(time_point.concs, pd.Series)
    assert isinstance(time_point.fluxes, pd.Series)
    assert time_point.concs.index.tolist() == ["S", "P"]
    assert time_point.fluxes.index.tolist() == ["v1", "v2"]


def test_timepoint_with_data(simple_model: Model) -> None:
    concs = pd.DataFrame({"S": [1.0, 2.0], "P": [3.0, 4.0]})
    fluxes = pd.DataFrame({"v1": [0.1, 0.2], "v2": [0.3, 0.4]})

    time_point = TimePoint.from_scan(simple_model, concs, fluxes, idx=1)
    assert time_point.concs["S"] == 2.0
    assert time_point.concs["P"] == 4.0
    assert time_point.fluxes["v1"] == 0.2
    assert time_point.fluxes["v2"] == 0.4


def test_timepoint_results(simple_model: Model) -> None:
    concs = pd.DataFrame({"S": [1.0], "P": [3.0]})
    fluxes = pd.DataFrame({"v1": [0.1], "v2": [0.3]})

    time_point = TimePoint.from_scan(simple_model, concs, fluxes, idx=0)
    results = time_point.results

    assert isinstance(results, pd.Series)
    assert len(results) == 4
    assert results["S"] == 1.0
    assert results["P"] == 3.0
    assert results["v1"] == 0.1
    assert results["v2"] == 0.3


def test_time_course_from_scan(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    time_course = TimeCourse.from_scan(simple_model, time_points, None, None)

    assert isinstance(time_course, TimeCourse)
    assert isinstance(time_course.concs, pd.DataFrame)
    assert isinstance(time_course.fluxes, pd.DataFrame)
    assert time_course.concs.shape == (3, 2)
    assert time_course.fluxes.shape == (3, 2)
    assert np.all(time_course.concs.index == time_points)
    assert np.all(time_course.fluxes.index == time_points)
    assert np.all(time_course.concs.columns == ["S", "P"])
    assert np.all(time_course.fluxes.columns == ["v1", "v2"])


def test_timecourse_with_data(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    concs = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=time_points, columns=["S", "P"]
    )
    fluxes = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], index=time_points, columns=["v1", "v2"]
    )

    time_course = TimeCourse.from_scan(simple_model, time_points, concs, fluxes)
    assert time_course.concs.loc[1.0, "S"] == 3.0  # type: ignore
    assert time_course.concs.loc[1.0, "P"] == 4.0  # type: ignore
    assert time_course.fluxes.loc[1.0, "v1"] == 0.3  # type: ignore
    assert time_course.fluxes.loc[1.0, "v2"] == 0.4  # type: ignore


def test_timecourse_results(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    concs = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=time_points, columns=["S", "P"]
    )
    fluxes = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], index=time_points, columns=["v1", "v2"]
    )

    time_course = TimeCourse.from_scan(simple_model, time_points, concs, fluxes)
    results = time_course.results

    assert isinstance(results, pd.DataFrame)
    assert results.shape == (3, 4)
    assert results.loc[1.0, "S"] == 3.0  # type: ignore
    assert results.loc[1.0, "P"] == 4.0  # type: ignore
    assert results.loc[1.0, "v1"] == 0.3  # type: ignore
    assert results.loc[1.0, "v2"] == 0.4  # type: ignore


def test_steady_state_worker(simple_model: Model) -> None:
    result = _steady_state_worker(simple_model, y0=None, rel_norm=False)
    assert isinstance(result, TimePoint)

    # The model should reach steady state with S=0, P=0
    assert not np.isnan(result.concs["S"])
    assert not np.isnan(result.concs["P"])
    assert not np.isnan(result.fluxes["v1"])
    assert not np.isnan(result.fluxes["v2"])


def test_time_course_worker(simple_model: Model) -> None:
    time_points = np.linspace(0, 1, 3)
    result = _time_course_worker(simple_model, y0=None, time_points=time_points)

    assert isinstance(result, TimeCourse)
    assert result.concs.shape == (3, 2)
    assert result.fluxes.shape == (3, 2)
    assert not np.isnan(result.concs.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_steady_state_scan(simple_model: Model) -> None:
    parameters = pd.DataFrame({"k1": [1.0, 2.0, 3.0]})

    result = steady_state(simple_model, parameters, parallel=False)

    assert result.concs.shape == (3, 2)
    assert result.fluxes.shape == (3, 2)
    assert result.parameters.equals(parameters)
    assert not np.isnan(result.concs.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_steady_state_scan_with_multiindex(simple_model: Model) -> None:
    parameters = pd.DataFrame({"k1": [1.0, 2.0], "k2": [3.0, 4.0]})

    result = steady_state(simple_model, parameters, parallel=False)

    assert result.concs.shape == (2, 2)
    assert result.fluxes.shape == (2, 2)
    assert isinstance(result.concs.index, pd.MultiIndex)
    assert isinstance(result.fluxes.index, pd.MultiIndex)
    assert not np.isnan(result.concs.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_time_course_scan(simple_model: Model) -> None:
    parameters = pd.DataFrame({"k1": [1.0, 2.0]})
    time_points = np.linspace(0, 1, 3)

    result = time_course(simple_model, parameters, time_points, parallel=False)

    assert result.concs.shape == (6, 2)  # 2 params x 3 time points x 2 variables
    assert result.fluxes.shape == (6, 2)  # 2 params x 3 time points x 2 reactions
    assert isinstance(result.concs.index, pd.MultiIndex)
    assert isinstance(result.fluxes.index, pd.MultiIndex)
    assert result.concs.index.names == ["n", "time"]
    assert result.fluxes.index.names == ["n", "time"]
    assert not np.isnan(result.concs.values).any()
    assert not np.isnan(result.fluxes.values).any()
