from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from modelbase2.types import Array


@dataclass
class Distribution(ABC):
    @abstractmethod
    def sample(self, num: int) -> Array: ...


@dataclass
class Beta(Distribution):
    a: float
    b: float
    seed: int = 42

    def sample(self, num: int) -> Array:
        return np.random.default_rng(seed=self.seed).beta(self.a, self.b, num)


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
class Normal(Distribution):
    loc: float
    scale: float
    seed: int = 42

    def sample(self, num: int) -> Array:
        return np.random.default_rng(seed=self.seed).normal(self.loc, self.scale, num)


@dataclass
class LogNormal(Distribution):
    mean: float
    sigma: float
    seed: int = 42

    def sample(self, num: int) -> Array:
        return np.random.default_rng(seed=self.seed).lognormal(
            self.mean, self.sigma, num
        )


@dataclass
class Skewnorm(Distribution):
    loc: float
    scale: float
    a: float

    def sample(self, num: int) -> Array:
        return cast(
            Array, stats.skewnorm(self.a, loc=self.loc, scale=self.scale).rvs(num)
        )


def sample(parameters: dict[str, Distribution], n: int) -> pd.DataFrame:
    return pd.DataFrame({k: v.sample(n) for k, v in parameters.items()})
