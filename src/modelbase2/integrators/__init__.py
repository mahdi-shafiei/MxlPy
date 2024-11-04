from __future__ import annotations

__all__ = [
    "Assimulo",
    "Scipy",
]


from .int_scipy import Scipy

try:
    from .int_assimulo import Assimulo

    DefaultIntegrator = Assimulo
except ImportError:
    DefaultIntegrator = Scipy
