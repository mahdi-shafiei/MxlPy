from modelbase2.types import ArrayLike, Callable, Protocol


class IntegratorProtocol(Protocol):
    """Interface for integrators"""

    def __init__(
        self,
        rhs: Callable,
        y0: ArrayLike,
    ) -> None: ...

    def reset(self) -> None: ...

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
