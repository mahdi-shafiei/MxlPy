"""Probability Distribution Classes for Parameter Sampling.

This module provides a collection of probability distributions used for parameter sampling
in metabolic modeling and Monte Carlo simulations.

Classes:
    Distribution (Protocol): Base protocol for all distribution classes
    Beta: Beta distribution for parameters bounded between 0 and 1
    Uniform: Uniform distribution for parameters with simple bounds
    Normal: Normal (Gaussian) distribution for unbounded parameters
    LogNormal: Log-normal distribution for strictly positive parameters
    Skewnorm: Skewed normal distribution for asymmetric parameter distributions

Each distribution class provides:
    - Consistent interface through the sample() method
    - Optional random number generator (RNG) control
    - Reproducible results via seed parameter

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats

from modelbase2.types import Array

__all__ = [
    "Beta",
    "Distribution",
    "LogNormal",
    "Normal",
    "RNG",
    "Skewnorm",
    "Uniform",
    "sample",
]

RNG = np.random.default_rng(seed=42)


class Distribution(Protocol):
    """Protocol defining interface for distribution classes.

    All distribution classes must implement the sample() method.
    """

    def sample(
        self,
        num: int,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Generate random samples from the distribution.

        Args:
            num: Number of samples to generate
            rng: Random number generator

        Returns:
            Array of random samples

        """
        ...


@dataclass
class Beta:
    """Beta distribution for parameters bounded between 0 and 1.

    Args:
        a: Alpha shape parameter (>0)
        b: Beta shape parameter (>0)

    """

    a: float
    b: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the beta distribution.

        Args:
            num: Number of samples to generate
            rng: Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.beta(self.a, self.b, num)


@dataclass
class Uniform:
    """Uniform distribution for parameters with simple bounds.

    Args:
        lower_bound: Minimum value
        upper_bound: Maximum value

    """

    lower_bound: float
    upper_bound: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the uniform distribution.

        Args:
            num: Number of samples to generate
            rng: Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.uniform(self.lower_bound, self.upper_bound, num)


@dataclass
class Normal:
    """Normal (Gaussian) distribution for unbounded parameters.

    Args:
        loc: Mean of the distribution
        scale: Standard deviation

    """

    loc: float
    scale: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the normal distribution.

        Args:
            num: Number of samples to generate
            rng: Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.normal(self.loc, self.scale, num)


@dataclass
class LogNormal:
    """Log-normal distribution for strictly positive parameters.

    Args:
        mean: Mean of the underlying normal distribution
        sigma: Standard deviation of the underlying normal distribution
        seed: Random seed for reproducibility

    """

    mean: float
    sigma: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the log-normal distribution.

        Args:
            num: Number of samples to generate
            rng: Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.lognormal(self.mean, self.sigma, num)


@dataclass
class Skewnorm:
    """Skewed normal distribution for asymmetric parameter distributions.

    Args:
        loc: Mean of the distribution
        scale: Standard deviation
        a: Skewness parameter

    """

    loc: float
    scale: float
    a: float

    def sample(
        self,
        num: int,
        rng: np.random.Generator | None = None,  # noqa: ARG002
    ) -> Array:
        """Generate random samples from the skewed normal distribution.

        Args:
            num: Number of samples to generate
            rng: The random generator arguemnt is unused but required for API compatibility

        """
        return cast(
            Array, stats.skewnorm(self.a, loc=self.loc, scale=self.scale).rvs(num)
        )


def sample(
    parameters: dict[str, Distribution],
    n: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate samples from the specified distributions.

    Args:
        parameters: Dictionary mapping parameter names to distribution objects.
        n: Number of samples to generate.
        rng: Random number generator.

    Returns:
        pd.DataFrame: DataFrame containing the generated samples.

    """
    return pd.DataFrame({k: v.sample(n, rng) for k, v in parameters.items()})


if __name__ == "__main__":
    _ = sample(
        {
            "beta": Beta(a=1.0, b=1.0),
            "uniform": Uniform(lower_bound=0.0, upper_bound=1.0),
            "normal": Normal(loc=1.0, scale=0.1),
            "log_normal": LogNormal(mean=1.0, sigma=0.1),
            "skewnorm": Skewnorm(loc=1.0, scale=0.1, a=5.0),
        },
        n=1,
    )
