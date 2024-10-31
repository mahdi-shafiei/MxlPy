import numpy as np
import pandas as pd

from modelbase2 import ModelProtocol
from modelbase2.scans._helpers import TimeCourse, TimePoint
from modelbase2.simulator import Simulator
from modelbase2.types import Array


def _steady_state_worker(
    model: ModelProtocol, y0: dict[str, float] | None
) -> TimePoint:
    c, v = (
        Simulator(model, y0=y0)
        .simulate_to_steady_state_and()
        .get_full_concs_and_fluxes()
    )
    return TimePoint(model, c, v)


def _time_course_worker(
    model: ModelProtocol, time_points: Array, y0: dict[str, float] | None
) -> TimeCourse:
    c, v = (
        Simulator(model, y0=y0)
        .simulate_and(time_points=time_points)
        .get_full_concs_and_fluxes()
    )
    return TimeCourse(model, time_points, c, v)


@staticmethod
def _parameter_scan_worker(
    parameter_value: float,
    *,
    parameter_name: str,
    model: RATE_MODEL_TYPE,
    sim: type[_BaseRateSimulator],
    integrator: type[AbstractIntegrator],
    tolerance: float,
    y0: ArrayLike,
    integrator_kwargs: dict[str, Any],
    include_fluxes: bool,
    rel_norm: bool,
) -> tuple[float, pd.Series, pd.Series]:
    m = model.copy()
    s = sim(model=m, integrator=integrator)
    s.initialise(y0=y0, test_run=False)
    s.update_parameter(
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )

    if (
        s.simulate_to_steady_state(
            tolerance=tolerance,
            rel_norm=rel_norm,
            **integrator_kwargs,
        )
        is None
    ):
        return parameter_value, _empty_conc_series(model), _empty_flux_series(model)

    if (full_results := s.get_full_results(concatenated=True)) is None:
        return parameter_value, _empty_conc_series(model), _empty_flux_series(model)
    last_full_results = full_results.iloc[-1]

    if include_fluxes:
        if (fluxes := s.get_fluxes(concatenated=True)) is None:
            return (
                parameter_value,
                _empty_conc_series(model),
                _empty_flux_series(model),
            )
        last_fluxes = fluxes.iloc[-1]
    else:
        last_fluxes = pd.Series()

    return parameter_value, last_full_results, last_fluxes


def parameter_scan(
    model: ModelProtocol,
    parameter_name: str,
    parameter_values: ArrayLike,
    tolerance: float = 1e-8,
    *,
    multiprocessing: bool = True,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    rel_norm: bool = False,
    **integrator_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan the model steady state changes caused by a change to a parameter."""
    if sys.platform in ["win32", "cygwin"]:
        warnings.warn(
            """
            Windows does not behave well with multiple processes.
            Falling back to threading routine.""",
            stacklevel=1,
        )
    worker = partial(
        _parameter_scan_worker,
        parameter_name=parameter_name,
        model=model,
        sim=__class__,
        integrator=_integrator,
        tolerance=tolerance,
        y0=y0,
        integrator_kwargs=integrator_kwargs,
        include_fluxes=True,
        rel_norm=rel_norm,
    )

    results: Iterable[tuple[float, pd.Series, pd.Series]]
    if sys.platform in ["win32", "cygwin"] or not multiprocessing:
        results = tqdm(
            map(worker, parameter_values),
            total=len(parameter_values),
            desc=parameter_name,
            disable=disable_tqdm,
        )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pe:
            results = tqdm(pe.map(worker, parameter_values))
    concentrations = {}
    fluxes = {}
    for i, conc, flux in results:
        concentrations[i] = conc
        fluxes[i] = flux
    return (
        pd.DataFrame(concentrations).T,
        pd.DataFrame(fluxes).T,
    )


def parameter_scan_2d(
    p1: tuple[str, ArrayLike],
    p2: tuple[str, ArrayLike],
    tolerance: float = 1e-8,
    *,
    disable_tqdm: bool = False,
    multiprocessing: bool = True,
    max_workers: int | None = None,
    rel_norm: bool = False,
    **integrator_kwargs: dict[str, Any],
) -> dict[float, pd.DataFrame]:
    cs = {}
    parameter_name1, parameter_values1 = p1
    parameter_name2, parameter_values2 = p2
    original_pars = model.get_parameters().copy()
    for value in tqdm(
        parameter_values2, total=len(parameter_values2), desc=parameter_name2
    ):
        update_parameter(parameter_name2, value)
        cs[value] = parameter_scan(
            parameter_name1,
            parameter_values1,
            tolerance=tolerance,
            disable_tqdm=disable_tqdm,
            multiprocessing=multiprocessing,
            max_workers=max_workers,
            rel_norm=rel_norm,
            **integrator_kwargs,
        )
    update_parameters(original_pars)
    return cs


def parameter_scan_2d_with_fluxes(
    self,
    p1: tuple[str, ArrayLike],
    p2: tuple[str, ArrayLike],
    tolerance: float = 1e-8,
    *,
    disable_tqdm: bool = False,
    multiprocessing: bool = True,
    max_workers: int | None = None,
    rel_norm: bool = False,
    **integrator_kwargs: dict[str, Any],
) -> tuple[dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    cs = {}
    vs = {}
    parameter_name1, parameter_values1 = p1
    parameter_name2, parameter_values2 = p2
    original_pars = model.get_parameters().copy()
    for value in tqdm(
        parameter_values2, total=len(parameter_values2), desc=parameter_name2
    ):
        update_parameter(parameter_name2, value)
        c, v = parameter_scan(
            parameter_name1,
            parameter_values1,
            tolerance=tolerance,
            multiprocessing=multiprocessing,
            disable_tqdm=disable_tqdm,
            max_workers=max_workers,
            rel_norm=rel_norm,
            **integrator_kwargs,  # type: ignore
        )
        cs[value] = c
        vs[value] = v
    update_parameters(original_pars)
    return cs, vs
