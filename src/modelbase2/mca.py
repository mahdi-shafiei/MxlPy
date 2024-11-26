"""Metabolic control analysis

Main functions:
    variable_elasticities
    parameter_elasticities
    response_coefficients
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from modelbase2.parallel import parallelise
from modelbase2.scans import _steady_state_worker
from modelbase2.types import ResponseCoefficients

if TYPE_CHECKING:
    from modelbase2.model import Model


_DISPLACEMENT = 1e-4


###############################################################################
# Non-steady state
###############################################################################


def variable_elasticities(
    model: Model,
    *,
    concs: dict[str, float] | None = None,
    variables: list[str] | None = None,
    time: float = 0,
    normalized: bool = True,
    displacement: float = _DISPLACEMENT,
) -> pd.DataFrame:
    """Get sensitivity of all rates to a change of the concentration of multiple compounds.

    Also called epsilon-elasticities. Not in steady state!
    """

    concs = model.get_initial_conditions() if concs is None else concs
    variables = model.get_variable_names() if variables is None else variables
    elasticities = {}

    for var in variables:
        old = concs[var]

        upper = model.get_fluxes(
            concs=concs | {var: old * (1 + displacement)}, time=time
        )
        lower = model.get_fluxes(
            concs=concs | {var: old * (1 - displacement)}, time=time
        )

        elasticity_coef = (upper - lower) / (2 * displacement * old)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(concs=concs, time=time)
        elasticities[var] = elasticity_coef

    return pd.DataFrame(data=elasticities)


def parameter_elasticities(
    model: Model,
    parameters: list[str] | None = None,
    concs: dict[str, float] | None = None,
    time: float = 0,
    *,
    normalized: bool = True,
    displacement: float = _DISPLACEMENT,
) -> pd.DataFrame:
    """Get sensitivity of all rates to a change of multiple parameter values.

    Also called pi-elasticities. Not in steady state!
    """
    concs = model.get_initial_conditions() if concs is None else concs
    parameters = model.get_parameter_names() if parameters is None else parameters

    elasticities = {}

    concs = model.get_initial_conditions() if concs is None else concs
    for par in parameters:
        old = model.parameters[par]

        model.update_parameters({par: old * (1 + displacement)})
        upper = model.get_fluxes(concs=concs, time=time)

        model.update_parameters({par: old * (1 - displacement)})
        lower = model.get_fluxes(concs=concs, time=time)

        # Reset
        model.update_parameters({par: old})
        elasticity_coef = (upper - lower) / (2 * displacement * old)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(concs=concs, time=time)
        elasticities[par] = elasticity_coef

    return pd.DataFrame(data=elasticities)


# ###############################################################################
# # Steady state
# ###############################################################################


def _response_coefficient_worker(
    parameter: str,
    *,
    model: Model,
    y0: dict[str, float] | None,
    normalized: bool,
    rel_norm: bool,
    displacement: float = _DISPLACEMENT,
) -> tuple[pd.Series, pd.Series]:
    old = model.parameters[parameter]

    model.update_parameters({parameter: old * (1 + displacement)})
    upper = _steady_state_worker(
        model,
        y0=y0,
        rel_norm=rel_norm,
    )

    model.update_parameters({parameter: old * (1 - displacement)})
    lower = _steady_state_worker(
        model,
        y0=y0,
        rel_norm=rel_norm,
    )

    conc_resp = (upper.concs - lower.concs) / (2 * displacement * old)
    flux_resp = (upper.fluxes - lower.fluxes) / (2 * displacement * old)
    # Reset
    model.update_parameters({parameter: old})
    if normalized:
        norm = _steady_state_worker(
            model,
            y0=y0,
            rel_norm=rel_norm,
        )
        conc_resp *= old / norm.concs
        flux_resp *= old / norm.fluxes
    return conc_resp, flux_resp


def response_coefficients(
    model: Model,
    parameters: list[str] | None = None,
    *,
    y0: dict[str, float] | None = None,
    normalized: bool = True,
    displacement: float = _DISPLACEMENT,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    rel_norm: bool = False,
) -> ResponseCoefficients:
    """Get response of the steady state concentrations and
    fluxes to a change of the given parameter.
    """
    parameters = model.get_parameter_names() if parameters is None else parameters

    res = parallelise(
        partial(
            _response_coefficient_worker,
            model=model,
            y0=y0,
            normalized=normalized,
            displacement=displacement,
            rel_norm=rel_norm,
        ),
        inputs=list(zip(parameters, parameters, strict=True)),
        cache=None,
        disable_tqdm=disable_tqdm,
        parallel=parallel,
        max_workers=max_workers,
    )
    return ResponseCoefficients(
        concs=pd.DataFrame({k: v[0] for k, v in res.items()}).T,
        fluxes=pd.DataFrame({k: v[1] for k, v in res.items()}).T,
    )
