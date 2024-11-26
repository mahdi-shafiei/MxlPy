"""modelbase2 is a package for creating and analysing metabolic models"""

from __future__ import annotations

__all__ = [
    "Assimulo",
    "Cache",
    "DefaultIntegrator",
    "Derived",
    "IntegratorProtocol",
    "LabelMapper",
    "LinearLabelMapper",
    "Model",
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
from .types import Derived, IntegratorProtocol

with contextlib.suppress(ImportError):
    from .integrators import Assimulo
