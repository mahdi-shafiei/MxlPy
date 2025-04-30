"""SBML support for mxlpy.

Allows importing and exporting metabolic models in SBML format.
"""

from __future__ import annotations

__all__ = [
    "read",
    "write",
]

from ._export import write
from ._import import read
