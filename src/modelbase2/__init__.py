from __future__ import annotations

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
    "utils",
]

import contextlib

from . import fit, surrogates, utils
from .integrators import DefaultIntegrator, IntegratorProtocol, Scipy
from .models import Model, ModelProtocol
from .simulator import Simulator
from .surrogates import TorchSurrogate

with contextlib.suppress(ImportError):
    from .integrators import Assimulo
