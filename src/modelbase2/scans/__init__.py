from __future__ import annotations

__all__ = [
    "Beta",
    "Skewnorm",
    "Normal",
    "Uniform",
    "LogNormal",
    "sample",
    "parallelise",
    "Cache",
    "steady_state_scan",
    "time_course",
    "time_course_over_protocol",
]

from ..parallel import Cache, parallelise
from ._dist import Beta, LogNormal, Normal, Skewnorm, Uniform, sample
from ._mc import (
    steady_state_scan,
    time_course,
    time_course_over_protocol,
)

if __name__ == "__main__":
    """Scans
    - Normal
        - steady-state over one parameters
        - steady-state over multiple parameters
        - time course over one parameter
        - time course over multiple parameters
        - protocol
    - Monte-Carlo distributed
        - steady-state over one parameters (+ Monte Carlo parameters)
        - steady-state over multiple parameters (+ Monte Carlo parameters)
        - time course over one parameter (+ Monte Carlo parameters)
        - time course over multiple parameters (+ Monte Carlo parameters)
        - protocol
    """
    import numpy as np

    from modelbase2 import Model
    from modelbase2.models.model_protocol import ModelProtocol
    from modelbase2.types import Array

    m = Model()

    def parameter_scan(model: ModelProtocol, parameters: dict[str, Array]): ...

    # Scan model steady-state over one parameter
    _ = parameter_scan(
        m,
        {
            "p": np.linspace(-5, 5, 11),
        },
    )

    # Scan model steady-state over multiple parameters
    _ = parameter_scan(
        m,
        {
            "p1": np.linspace(-5, 5, 11),
            "p2": np.linspace(-5, 5, 11),
        },
    )

    # S
