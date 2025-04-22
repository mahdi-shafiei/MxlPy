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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam

from modelbase2.parallel import Cache
from modelbase2.types import AbstractSurrogate

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

__all__ = [
    "Approximator",
    "DefaultCache",
    "DefaultDevice",
    "TorchSurrogate",
    "train_torch_surrogate",
]


DefaultDevice = torch.device("cpu")
DefaultCache = Cache(Path(".cache"))


@dataclass(kw_only=True)
class TorchSurrogate(AbstractSurrogate):
    """Surrogate model using PyTorch.

    Attributes:
        model: PyTorch neural network model.

    Methods:
        predict: Predict outputs based on input data using the PyTorch model.

    """

    model: torch.nn.Module

    def predict(self, y: np.ndarray) -> dict[str, float]:
        """Predict outputs based on input data using the PyTorch model.

        Args:
            y: Input data as a numpy array.

        Returns:
            dict[str, float]: Dictionary mapping output variable names to predicted values.

        """
        with torch.no_grad():
            return dict(
                zip(
                    self.stoichiometries,
                    self.model(
                        torch.tensor(y, dtype=torch.float32),
                    ).numpy(),
                    strict=True,
                )
            )


class Approximator(nn.Module):
    """Neural network approximator for surrogate modeling.

    Attributes:
        net: Sequential neural network model.

    Methods:
        forward: Forward pass through the neural network.

    """

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        """Initializes the surrogate model with the given number of inputs and outputs.

        Args:
            n_inputs (int): The number of input features.
            n_outputs (int): The number of output features.

        Initializes a neural network with the following architecture:
        - Linear layer with `n_inputs` inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and `n_outputs` outputs

        The weights of the linear layers are initialized with a normal distribution
        (mean=0, std=0.1) and the biases are initialized to 0.

        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, n_outputs),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.net(x)


def _train_batched(
    aprox: nn.Module,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    optimizer: Adam,
    device: torch.device,
    batch_size: int,
) -> pd.Series:
    """Train the neural network using mini-batch gradient descent.

    Args:
        aprox: Neural network model to train.
        features: Input features as a tensor.
        targets: Target values as a tensor.
        epochs: Number of training epochs.
        optimizer: Optimizer for training.
        device: torch device
        batch_size: Size of mini-batches for training.

    Returns:
        pd.Series: Series containing the training loss history.

    """
    rng = np.random.default_rng()
    losses = {}
    for i in tqdm.trange(epochs):
        idxs = rng.choice(features.index, size=batch_size)
        X = torch.Tensor(features.iloc[idxs].to_numpy(), device=device)
        Y = torch.Tensor(targets.iloc[idxs].to_numpy(), device=device)
        optimizer.zero_grad()
        loss = torch.mean(torch.abs(aprox(X) - Y))
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach().numpy()
    return pd.Series(losses, dtype=float)


def _train_full(
    aprox: nn.Module,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    optimizer: Adam,
    device: torch.device,
) -> pd.Series:
    """Train the neural network using full-batch gradient descent.

    Args:
        aprox: Neural network model to train.
        features: Input features as a tensor.
        targets: Target values as a tensor.
        epochs: Number of training epochs.
        optimizer: Optimizer for training.
        device: Torch device

    Returns:
        pd.Series: Series containing the training loss history.

    """
    X = torch.Tensor(features.to_numpy(), device=device)
    Y = torch.Tensor(targets.to_numpy(), device=device)

    losses = {}
    for i in tqdm.trange(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.abs(aprox(X) - Y))
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach().numpy()
    return pd.Series(losses, dtype=float)


def train_torch_surrogate(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    surrogate_inputs: list[str],
    surrogate_stoichiometries: dict[str, dict[str, float]],
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: Callable[[ParamsT], Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchSurrogate, pd.Series]:
    """Train a PyTorch surrogate model.

    Examples:
        >>> train_torch_surrogate(
        ...     features,
        ...     targets,
        ...     epochs=100,
        ...     surrogate_inputs=["x1", "x2"],
        ...     surrogate_stoichiometries={
        ...         "v1": {"x1": -1, "x2": 1, "ATP": -1},
        ...     },
        ...)

    Args:
        features: DataFrame containing the input features for training.
        targets: DataFrame containing the target values for training.
        epochs: Number of training epochs.
        surrogate_inputs: List of input variable names for the surrogate model.
        surrogate_stoichiometries: Dictionary mapping reaction names to stoichiometries.
        batch_size: Size of mini-batches for training (None for full-batch).
        approximator: Predefined neural network model (None to use default).
        optimimzer_cls: Optimizer class to use for training (default: Adam).
        device: Device to run the training on (default: DefaultDevice).

    Returns:
        tuple[TorchSurrogate, pd.Series]: Trained surrogate model and loss history.

    """
    if approximator is None:
        approximator = Approximator(
            n_inputs=len(features.columns),
            n_outputs=len(targets.columns),
        ).to(device)

    optimizer = optimimzer_cls(approximator.parameters())
    if batch_size is None:
        losses = _train_full(
            aprox=approximator,
            features=features,
            targets=targets,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
        )
    else:
        losses = _train_batched(
            aprox=approximator,
            features=features,
            targets=targets,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
        )
    surrogate = TorchSurrogate(
        model=approximator,
        args=surrogate_inputs,
        stoichiometries=surrogate_stoichiometries,
    )
    return surrogate, losses
