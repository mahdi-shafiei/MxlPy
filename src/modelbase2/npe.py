from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam

from modelbase2.parallel import Cache

DefaultDevice = torch.device("cpu")
DefaultCache = Cache(Path(".cache"))


class DefaultSSAproximator(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
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
        return self.net(x)


class DefaultTimeSeriesApproximator(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int) -> None:
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(n_inputs, n_hidden)
        self.to_out = nn.Linear(n_hidden, n_outputs)

        nn.init.normal_(self.to_out.weight, mean=0, std=0.1)
        nn.init.constant_(self.to_out.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out, (hidden_state, cell_state)
        _, (hn, _) = self.lstm(x)
        return self.to_out(hn[-1])  # Use last hidden state


@dataclass(kw_only=True)
class AbstractEstimator:
    parameter_names: list[str]

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(kw_only=True)
class TorchSSEstimator(AbstractEstimator):
    model: torch.nn.Module

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        with torch.no_grad():
            pred = self.model(torch.tensor(features.to_numpy(), dtype=torch.float32))
            return pd.DataFrame(pred, columns=self.parameter_names)


@dataclass(kw_only=True)
class TorchTimeSeriesEstimator(AbstractEstimator):
    model: torch.nn.Module

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        idx = cast(pd.MultiIndex, features.index)
        features_ = torch.Tensor(
            np.swapaxes(
                features.to_numpy().reshape(
                    (
                        len(idx.levels[0]),
                        len(idx.levels[1]),
                        len(features.columns),
                    )
                ),
                axis1=0,
                axis2=1,
            ),
        )
        with torch.no_grad():
            pred = self.model(features_)
            return pd.DataFrame(pred, columns=self.parameter_names)


def _train_batched(
    approximator: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    optimizer: Adam,
    batch_size: int,
) -> pd.Series:
    losses = {}

    for epoch in tqdm.trange(epochs):
        permutation = torch.randperm(features.size()[0])
        epoch_loss = 0
        for i in range(0, features.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i : i + batch_size]

            loss = torch.mean(
                torch.abs(approximator(features[indices]) - targets[indices])
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()

        losses[epoch] = epoch_loss / (features.size()[0] / batch_size)
    return pd.Series(losses, dtype=float)


def _train_full(
    approximator: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    optimizer: Adam,
) -> pd.Series:
    losses = {}
    for i in tqdm.trange(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.abs(approximator(features) - targets))
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach().numpy()
    return pd.Series(losses, dtype=float)


def train_torch_ss_estimator(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: type[Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchSSEstimator, pd.Series]:
    if approximator is None:
        approximator = DefaultSSAproximator(
            n_inputs=len(features.columns),
            n_outputs=len(targets.columns),
        ).to(device)

    features_ = torch.Tensor(features.to_numpy(), device=device)
    targets_ = torch.Tensor(targets.to_numpy(), device=device)

    optimizer = optimimzer_cls(approximator.parameters())
    if batch_size is None:
        losses = _train_full(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
        )
    else:
        losses = _train_batched(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
            batch_size=batch_size,
        )

    return TorchSSEstimator(
        model=approximator,
        parameter_names=list(targets.columns),
    ), losses


def train_torch_time_series_estimator(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: type[Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchTimeSeriesEstimator, pd.Series]:
    if approximator is None:
        approximator = DefaultTimeSeriesApproximator(
            n_inputs=len(features.columns),
            n_outputs=len(targets.columns),
            n_hidden=1,
        ).to(device)

    optimizer = optimimzer_cls(approximator.parameters())
    features_ = torch.Tensor(
        np.swapaxes(
            features.to_numpy().reshape((len(targets), -1, len(features.columns))),
            axis1=0,
            axis2=1,
        ),
        device=device,
    )
    targets_ = torch.Tensor(targets.to_numpy(), device=device)
    if batch_size is None:
        losses = _train_full(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
        )
    else:
        losses = _train_batched(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
            batch_size=batch_size,
        )
    return TorchTimeSeriesEstimator(
        model=approximator,
        parameter_names=list(targets.columns),
    ), losses
