"""Surrogate Models Module.

This module provides classes and functions for creating and training surrogate models
for metabolic simulations. It includes functionality for both steady-state and time-series
data using neural networks.

Classes:
    AbstractSurrogate: Abstract base class for surrogate models.
    TorchSurrogate: Surrogate model using PyTorch.
    Approximator: Neural network approximator for surrogate modeling.

Functions:
    train_torch_surrogate: Train a PyTorch surrogate model.
    train_torch_time_course_estimator: Train a PyTorch time course estimator.
"""

from __future__ import annotations

import contextlib
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from modelbase2.parallel import Cache

if TYPE_CHECKING:
    import numpy as np

with contextlib.suppress(ImportError):
    from ._torch import Dense, TorchSurrogate, train_torch_surrogate

__all__ = [
    "AbstractSurrogate",
    "DefaultCache",
    "Dense",
    "MockSurrogate",
    "TorchSurrogate",
    "train_torch_surrogate",
]


DefaultCache = Cache(Path(".cache"))


@dataclass(kw_only=True)
class AbstractSurrogate:
    """Abstract base class for surrogate models.

    Attributes:
        inputs: List of input variable names.
        stoichiometries: Dictionary mapping reaction names to stoichiometries.

    Methods:
        predict: Abstract method to predict outputs based on input data.

    """

    args: list[str] = field(default_factory=list)
    stoichiometries: dict[str, dict[str, float]] = field(default_factory=dict)

    @abstractmethod
    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict outputs based on input data."""

    def predict(self, y: np.ndarray) -> dict[str, float]:
        """Predict outputs based on input data."""
        return dict(
            zip(
                self.stoichiometries,
                self.predict_raw(y),
                strict=True,
            )
        )


@dataclass(kw_only=True)
class MockSurrogate(AbstractSurrogate):
    """Mock surrogate model for testing purposes."""

    def predict_raw(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """Predict outputs based on input data."""
        return y
