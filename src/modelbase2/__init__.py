from __future__ import annotations

from .types import IntegratorProtocol, ModelProtocol

__all__ = [
    "Assimulo",
    "DefaultIntegrator",
    "IntegratorProtocol",
    "LabelMapper",
    "LinearLabelMapper",
    "Model",
    "ModelProtocol",
    "Scipy",
    "Simulator",
    "TorchSurrogate",
    "distributions",
    "fit",
    "mc",
    "mca",
    "parameter_scan_ss",
    "parameter_scan_time_series",
    "plot",
    "surrogates",
]

import contextlib

from . import distributions, fit, mc, mca, plot
from .integrators import DefaultIntegrator, Scipy
from .label_map import LabelMapper
from .linear_label_map import LinearLabelMapper
from .model import Model
from .scans import parameter_scan_ss, parameter_scan_time_series
from .simulator import Simulator
from .surrogates import TorchSurrogate

# from . import sbml

with contextlib.suppress(ImportError):
    from .integrators import Assimulo
