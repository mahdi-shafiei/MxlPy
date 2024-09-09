from __future__ import annotations

import multiprocessing
import pickle  # nosec
import sys
from abc import ABC, abstractmethod
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Hashable,
    Iterable,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
import pebble
from scipy import stats
from tqdm import tqdm

from modelbase.ode import Model, Simulator

if TYPE_CHECKING:
    from modelbase.typing import Array, Axis

T = TypeVar("T")
V = TypeVar("V")
Tin = TypeVar("Tin")
Tout = TypeVar("Tout")
Ti = TypeVar("Ti", bound=Iterable)
K = TypeVar("K", bound=Hashable)


@dataclass
class Distribution(ABC):
    @abstractmethod
    def sample(self, num: int) -> Array: ...


@dataclass
class Uniform(Distribution):
    lower_bound: float
    upper_bound: float
    seed: int = 42

    def sample(self, num: int) -> Array:
        return np.random.default_rng(seed=self.seed).uniform(
            self.lower_bound, self.upper_bound, num
        )


@dataclass
class Skewnorm(Distribution):
    loc: float
    scale: float
    a: float

    def sample(self, num: int) -> Array:
        return stats.skewnorm(self.a, loc=self.loc, scale=self.scale).rvs(num)


def mc_sample(parameters: dict[str, Distribution], n: int) -> pd.DataFrame:
    return pd.DataFrame({k: v.sample(n) for k, v in parameters.items()})


def _pickle_name(k: Hashable) -> str:
    return f"{k}.p"


def _pickle_load(file: Path) -> Any:
    with file.open("rb") as fp:
        return pickle.load(fp)  # nosec


def _pickle_save(file: Path, data: Any) -> None:
    with file.open("wb") as fp:
        pickle.dump(data, fp)


@dataclass
class Cache:
    tmp_dir: Path = Path(".cache")
    name_fn: Callable[[Any], str] = _pickle_name
    load_fn: Callable[[Path], Any] = _pickle_load
    save_fn: Callable[[Path, Any], None] = _pickle_save


def _default_if_none(el: T | None, default: T) -> T:
    return default if el is None else el


def _load_or_run(
    inp: tuple[K, Tin],
    fn: Callable[[Tin], Tout],
    cache: Cache | None,
) -> tuple[K, Tout]:
    k, v = inp
    if cache is None:
        res = fn(v)
    else:
        file = cache.tmp_dir / cache.name_fn(k)
        if file.exists():
            return k, cast(Tout, cache.load_fn(file))
        res = fn(v)
        cache.save_fn(file, res)
    return k, res


def parallelise(
    fn: Callable[[Tin], Tout],
    inputs: Collection[tuple[K, Tin]],
    *,
    cache: Cache | None,
    max_workers: int | None = None,
    timeout: float | None = None,
    disable_tqdm: bool = False,
    tqdm_desc: str | None = None,
) -> dict[Tin, Tout]:
    if cache is not None:
        cache.tmp_dir.mkdir(parents=True, exist_ok=True)

    worker: Callable[[K, Tin], tuple[K, Tout]] = partial(
        _load_or_run,
        fn=fn,
        cache=cache,
    )  # type: ignore

    results: dict[Tin, Tout]
    if sys.platform in ["win32", "cygwin"]:
        results = dict(
            tqdm(
                map(worker, inputs),  # type: ignore
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            )  # type: ignore
        )  # type: ignore
    else:
        results = {}
        max_workers = _default_if_none(max_workers, multiprocessing.cpu_count())

        with tqdm(
            total=len(inputs),
            disable=disable_tqdm,
            desc=tqdm_desc,
        ) as pbar, pebble.ProcessPool(max_workers=max_workers) as pool:
            future = pool.map(worker, inputs, timeout=timeout)
            it = future.result()
            while True:
                try:
                    key, value = next(it)
                    pbar.update(1)
                    results[key] = value
                except StopIteration:  # noqa: PERF203
                    break
                except TimeoutError:
                    pbar.update(1)
    return results


def _empty_conc_series(model: Model) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.compounds), fill_value=np.nan),
        index=model.compounds,
    )


def _empty_flux_series(model: Model) -> pd.Series:
    return pd.Series(
        data=np.full(shape=len(model.rates), fill_value=np.nan),
        index=model.rates,
    )


def _empty_conc_df(model: Model, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.compounds)),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.compounds,
    )


def _empty_flux_df(model: Model, time_points: Array) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.rates)),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.rates,
    )


def empty_time_point(model: Model) -> tuple[pd.Series, pd.Series]:
    return _empty_conc_series(model), _empty_flux_series(model)


def empty_time_course(
    model: Model, time_points: Array
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _empty_conc_df(model, time_points), _empty_flux_df(model, time_points)


def _time_course_worker(
    pars: pd.Series,
    model: Model,
    y0: dict[str, float],
    time_points: Array,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c, v = (
        Simulator(model)
        .initialise(y0)
        .update_parameters(pars.to_dict())
        .simulate_and(time_points=time_points)
        .get_full_results_and_fluxes_df()
    )
    if c is None or v is None:
        return empty_time_course(model, time_points)
    return c, v


def time_course(
    model: Model,
    y0: dict[str, float],
    time_points: Array,
    mc_parameters: pd.DataFrame,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = parallelise(
        partial(
            _time_course_worker,
            model=model,
            y0=y0,
            time_points=time_points,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v[0] for k, v in res.items()}
    fluxes = {k: v[1] for k, v in res.items()}
    return pd.DataFrame(concs), pd.DataFrame(fluxes)


def _time_course_over_protocol_worker(
    pars: pd.Series,
    model: Model,
    y0: dict[str, float],
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_points = np.linspace(
        0,
        protocol.index[-1].total_seconds(),
        len(protocol) * time_points_per_step,
    )
    s = Simulator(model).initialise(y0).update_parameters(pars.to_dict())
    for t_end, ser in protocol.iterrows():
        t_end = cast(pd.Timedelta, t_end)
        s.update_parameters(ser.to_dict())
        s.simulate(t_end.total_seconds(), steps=time_points_per_step)
    c, v = s.get_full_results_and_fluxes_df()
    if c is None or v is None:
        return empty_time_course(model, time_points)
    return c, v


def time_course_over_protocol(
    model: Model,
    y0: dict[str, float],
    protocol: pd.DataFrame,
    mc_parameters: pd.DataFrame,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = parallelise(
        partial(
            _time_course_over_protocol_worker,
            model=model,
            y0=y0,
            protocol=protocol,
            time_points_per_step=time_points_per_step,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v[0] for k, v in res.items()}
    fluxes = {k: v[1] for k, v in res.items()}
    return pd.DataFrame(concs), pd.DataFrame(fluxes)


def _steady_state_scan_worker(
    pars: pd.Series,
    model: Model,
    y0: dict[str, float],
    parameter: str,
    parameter_values: Array,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        Simulator(model)
        .initialise(y0)
        .update_parameters(pars.to_dict())
        .parameter_scan_with_fluxes(parameter, parameter_values)
    )


def steady_state_scan(
    model: Model,
    y0: dict[str, float],
    parameter: str,
    parameter_values: Array,
    mc_parameters: pd.DataFrame,
    max_workers: int | None = None,
    cache: Cache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = parallelise(
        partial(
            _steady_state_scan_worker,
            model=model,
            y0=y0,
            parameter=parameter,
            parameter_values=parameter_values,
        ),
        inputs=list(mc_parameters.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    concs = {k: v[0] for k, v in res.items()}
    fluxes = {k: v[1] for k, v in res.items()}
    return pd.DataFrame(concs), pd.DataFrame(fluxes)


def plot_line_mean_std(
    ax: Axis,
    df: pd.DataFrame,
    color: str,
    label: str,
    alpha: float = 0.2,
) -> None:
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ax.plot(mean, color=color, label=label)
    ax.fill_between(
        df.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )


def plot_line_median_std(
    ax: Axis,
    cpd: pd.DataFrame,
    color: str,
    label: str,
    alpha: float = 0.2,
) -> None:
    mean = cpd.median(axis=1)
    std = cpd.std(axis=1)
    ax.plot(mean, color=color, label=label)
    ax.fill_between(
        cpd.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )
