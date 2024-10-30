from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Assimulo",
]

from typing import TYPE_CHECKING, Literal

import numpy as np
from assimulo.problem import Explicit_Problem  # type: ignore
from assimulo.solvers import CVode  # type: ignore
from assimulo.solvers.sundials import CVodeError  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelbase2.types import ArrayLike


@dataclass
class Assimulo:
    rhs: Callable
    y0: ArrayLike
    atol: float = 1e-8
    rtol: float = 1e-8
    maxnef: int = 4  # max error failures
    maxncf: int = 1  # max convergence failures
    verbosity: Literal[50, 40, 30, 20, 10] = 50

    def __post_init__(self) -> None:
        self.integrator = CVode(Explicit_Problem(self.rhs, self.y0))
        self.integrator.atol = self.atol
        self.integrator.rtol = self.rtol
        self.integrator.maxnef = self.maxnef
        self.integrator.maxncf = self.maxncf
        self.integrator.verbosity = self.verbosity

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]:
        if steps is None:
            steps = 0
        try:
            return self.integrator.simulate(t_end, steps, time_points)  # type: ignore
        except CVodeError:
            return None, None

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        t_max: float = 1_000_000_000,
    ) -> tuple[float | None, ArrayLike | None]:
        self.integrator.reset()

        try:
            for t_end in np.geomspace(1000, t_max, 3):
                t, y = self.integrator.simulate(t_end)
                diff = (y[-1] - y[-2]) / y[-1] if rel_norm else y[-1] - y[-2]
                if np.linalg.norm(diff, ord=2) < tolerance:
                    return t[-1], y[-1]
        except CVodeError:
            return None, None
        return None, None
