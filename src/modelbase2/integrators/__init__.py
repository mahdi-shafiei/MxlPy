"""Integrator Package.

This package provides integrators for solving ordinary differential equations (ODEs).
It includes support for both Assimulo and Scipy integrators, with Assimulo being the default if available.
"""

from __future__ import annotations

__all__ = ["DefaultIntegrator"]


from .int_scipy import Scipy

try:
    from .int_assimulo import Assimulo

    DefaultIntegrator = Assimulo
except ImportError:
    DefaultIntegrator = Scipy
