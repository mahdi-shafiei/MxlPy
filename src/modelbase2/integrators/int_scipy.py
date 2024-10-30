from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Scipy",
]

import copy
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.integrate as spi

from modelbase2.types import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Scipy:
    rhs: Callable
    y0: ArrayLike
    atol: float = 1e-8
    rtol: float = 1e-8
    t0: float = 0.0
    _y0_orig: ArrayLike = field(default_factory=list)

    def __post_init__(self) -> None:
        self._y0_orig = self.y0.copy()

    def reset(self) -> None:
        self.t0 = 0
        self.y0 = self._y0_orig.copy()

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]:
        if time_points is not None:
            if time_points[0] != 0:
                t = [self.t0]
                t.extend(time_points)
            else:
                t = cast(list, time_points)
            t_array = np.array(t)
        elif steps is not None and t_end is not None:
            # Scipy counts the total amount of return points rather than
            # steps as assimulo
            steps += 1
            t_array = np.linspace(self.t0, t_end, steps)
        elif t_end is not None:
            t_array = np.linspace(self.t0, t_end, 100)
        else:
            msg = "You need to supply t_end (+steps) or time_points"
            raise ValueError(msg)

        y = spi.odeint(
            func=self.rhs,
            y0=self.y0,
            t=t_array,
            tfirst=True,
            atol=self.atol,
            rtol=self.rtol,
        )
        self.t0 = t_array[-1]
        self.y0 = y[-1, :]
        return t_array, y

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        step_size: int = 100,
        max_steps: int = 1000,
        integrator: str = "lsoda",
    ) -> tuple[float | None, ArrayLike | None]:
        self.reset()
        integ = spi.ode(self.rhs)
        integ.set_integrator(
            name=integrator,
            step_size=step_size,
            max_steps=max_steps,
            integrator=integrator,
        )
        integ.set_initial_value(self.y0)
        t = self.t0 + step_size
        y1 = copy.deepcopy(self.y0)
        for _ in range(max_steps):
            y2 = integ.integrate(t)
            diff = (y2 - y1) / y1 if rel_norm else y2 - y1
            if np.linalg.norm(diff, ord=2) < tolerance:
                return t, cast(ArrayLike, y2)
            y1 = y2
            t += step_size
        return None, None
