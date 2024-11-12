from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam

from modelbase2 import Simulator
from modelbase2.parallel import Cache, parallelise
from modelbase2.scans import _empty_flux_series
from modelbase2.types import ModelProtocol

if TYPE_CHECKING:
    import numpy as np

DefaultDevice = torch.device("cpu")
DefaultCache = Cache(Path(".cache"))


@dataclass(kw_only=True)
class AbstractEstimator:
    parameter_names: list[str]

    @abstractmethod
    def predict(self, y: np.ndarray) -> pd.DataFrame: ...


@dataclass(kw_only=True)
class TorchEstimator(AbstractEstimator):
    model: torch.nn.Module

    def predict(self, y: np.ndarray) -> pd.DataFrame:
        with torch.no_grad():
            pred = self.model(torch.tensor(y, dtype=torch.float32))
            return pd.DataFrame(pred, columns=self.parameter_names)


class DefaultApproximator(nn.Module):
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


def _ss_flux(
    params: pd.Series,
    model: ModelProtocol,
) -> pd.Series:
    flux = (
        Simulator(model.update_parameters(params.to_dict()))
        .simulate_to_steady_state()
        .get_fluxes()
    )
    if flux is None:
        return _empty_flux_series(model)
    return flux.iloc[-1]


def create_ss_flux_data(
    model: ModelProtocol,
    parameters: pd.DataFrame,
    cache: Cache | None = DefaultCache,
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        (
            pd.concat(
                parallelise(
                    partial(
                        _ss_flux,
                        model=model,
                    ),
                    inputs=list(parameters.iterrows()),
                    cache=cache,
                )
            )
            .unstack()
            .fillna(0)
        ),
    )


def _train_batched(
    aprox: nn.Module,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    optimizer: Adam,
    device: torch.device,
    batch_size: int,
) -> pd.Series:
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


def train_torch_estimator(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: type[Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchEstimator, pd.Series]:
    if approximator is None:
        approximator = DefaultApproximator(
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
    surrogate = TorchEstimator(
        model=approximator,
        parameter_names=list(targets.columns),
    )
    return surrogate, losses
