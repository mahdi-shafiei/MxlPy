"""modelbase2: A Python Package for Metabolic Modeling and Analysis.

This package provides tools for creating, simulating and analyzing metabolic models
with features including:

Key Features:
    - Model creation and manipulation
    - Steady state and time-series simulations
    - Parameter fitting and optimization
    - Monte Carlo analysis
    - Metabolic Control Analysis (MCA)
    - SBML import/export support
    - Visualization tools

Core Components:
    Model: Core class for metabolic model representation
    Simulator: Handles model simulation and integration
    DefaultIntegrator: Standard ODE solver implementation
    LabelMapper: Maps between model components and labels
    Cache: Performance optimization through result caching

Simulation Features:
    - Steady state calculations
    - Time course simulations
    - Parameter scanning
    - Surrogate modeling with PyTorch

Analysis Tools:
    - Parameter fitting to experimental data
    - Monte Carlo methods for uncertainty analysis
    - Metabolic Control Analysis
    - Custom visualization functions

"""

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
