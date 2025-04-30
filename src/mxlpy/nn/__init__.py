"""Collection of neural network architectures."""

import contextlib

__all__ = ["tensorflow", "torch"]

with contextlib.suppress(ImportError):
    from . import _torch as torch

from . import _tensorflow as tensorflow
