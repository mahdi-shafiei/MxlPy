from __future__ import annotations

__all__ = [
    "Integrator",
    "Assimulo",
    "Scipy",
]

import contextlib

with contextlib.suppress(ImportError):
    from .int_assimulo import Assimulo
from .int_scipy import Scipy
from .protocol import Integrator
