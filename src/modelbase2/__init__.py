from __future__ import annotations

from .types import IntegratorProtocol, ModelProtocol

__all__ = [
    "Assimulo",
    "Cache",
    "DefaultIntegrator",
    "IntegratorProtocol",
    "LabelMapper",
    "LinearLabelMapper",
    "Model",
    "ModelProtocol",
    "Scipy",
    "Simulator",
    "TorchSurrogate",
    "cartesian_product",
    "distributions",
    "fit",
    "mc",
    "mca",
    "parameter_scan_protocol",
    "parameter_scan_ss",
    "parameter_scan_time_series",
    "plot",
    "sbml",
    "surrogates",
]

import contextlib

from . import distributions, fit, mc, mca, plot, sbml
from .integrators import DefaultIntegrator, Scipy
from .label_map import LabelMapper
from .linear_label_map import LinearLabelMapper
from .mc import Cache
from .model import Model
from .scans import (
    cartesian_product,
    parameter_scan_protocol,
    parameter_scan_ss,
    parameter_scan_time_series,
)
from .simulator import Simulator
from .surrogates import TorchSurrogate

with contextlib.suppress(ImportError):
    from .integrators import Assimulo
