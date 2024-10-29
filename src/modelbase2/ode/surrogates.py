from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


@dataclass(kw_only=True)
class AbstractSurrogate:
    inputs: list[str]
    fluxes: list[str]
    stoichiometries: dict[str, dict[str, float]]

    @abstractmethod
    def predict(self, y: np.ndarray) -> dict[str, float]: ...


@dataclass(kw_only=True)
class TorchSurrogate(AbstractSurrogate):
    model: torch.nn.Module

    def predict(self, y: np.ndarray) -> dict[str, float]:
        with torch.no_grad():
            return dict(
                zip(
                    self.fluxes,
                    self.model(
                        torch.tensor(y, dtype=torch.float32),
                    ).numpy(),
                )
            )
