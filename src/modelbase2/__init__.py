from __future__ import annotations

from .types import IntegratorProtocol, ModelProtocol

__all__ = [
    "Assimulo",
    "DefaultIntegrator",
    "IntegratorProtocol",
    "Model",
    "ModelProtocol",
    "Scipy",
    "Simulator",
    "TorchSurrogate",
    "fit",
    "surrogates",
    "parameter_scan_ss",
    "parameter_scan_time_series",
    "mc",
    "mca",
    "plot",
    "distributions",
]

import contextlib

from . import distributions, fit, mc, mca, plot, surrogates
from .integrators import DefaultIntegrator, Scipy
from .models import LabelModel, LinearLabelModel, Model
from .scans import parameter_scan_ss, parameter_scan_time_series
from .simulator import Simulator
from .surrogates import TorchSurrogate

# from . import sbml

with contextlib.suppress(ImportError):
    from .integrators import Assimulo

if __name__ == "__main__":
    _ = Simulator(Model()).simulate(10)
    # _ = Simulator(LabelModel()).simulate(10)
    # _ = Simulator(LinearLabelModel()).simulate(10)
