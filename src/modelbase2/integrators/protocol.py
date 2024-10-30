from typing import Protocol

from modelbase2.types import ArrayLike


class Integrator(Protocol):
    """Interface for integrators"""

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]: ...

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> tuple[float | None, ArrayLike | None]: ...
