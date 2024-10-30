from __future__ import annotations

__all__ = [
    "Assimulo",
    "IntegratorProtocol",
    "Scipy",
]


from .int_scipy import Scipy

try:
    from .int_assimulo import Assimulo

    DefaultIntegrator = Assimulo
except ImportError:
    DefaultIntegrator = Scipy


from .protocol import IntegratorProtocol
