"""Metabolic Control Analysis (MCA) Module.

Provides functions for analyzing control and regulation in metabolic networks through:
- Elasticity coefficients (variable and parameter)
- Response coefficients

Main Functions:
    variable_elasticities: Calculate non-steady state variable elasticities
    parameter_elasticities: Calculate non-steady state parameter elasticities
    response_coefficients: Calculate response coefficients for steady state

Mathematical Background:
    MCA quantifies how changes in system parameters affect metabolic variables
    through elasticity and control coefficients. These describe the sensitivity
    of reaction rates and steady state variables to perturbations.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from mxlpy.integrators import DefaultIntegrator
from mxlpy.parallel import parallelise
from mxlpy.scan import _steady_state_worker
from mxlpy.types import ArrayLike, ResponseCoefficients

__all__ = [
    "parameter_elasticities",
    "response_coefficients",
    "variable_elasticities",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy import IntegratorProtocol
    from mxlpy.model import Model


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
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate non-steady state elasticity coefficients.

    Computes the sensitivity of reaction rates to changes in metabolite
    concentrations (ε-elasticities).

    Examples:
        >>> variable_elasticities(model, concs={"A": 1.0, "B": 2.0})
        Rxn     A     B
         v1   0.0   0.0
         v2   1.0   0.0
         v3   0.0   5.0


    Args:
        model: Metabolic model instance
        concs: Initial concentrations {metabolite: value}. Uses model defaults if None
        variables: List of variables to analyze. Uses all if None
        time: Time point for evaluation
        normalized: Whether to normalize coefficients
        displacement: Relative perturbation size

    Returns:
        DataFrame with elasticity coefficients (reactions x metabolites)

    """
    concs = model.get_initial_conditions() if concs is None else concs
    variables = model.get_variable_names() if variables is None else variables
    elasticities = {}

    for var in variables:
        old = concs[var]

        upper = model.get_fluxes(
            variables=concs | {var: old * (1 + displacement)}, time=time
        )
        lower = model.get_fluxes(
            variables=concs | {var: old * (1 - displacement)}, time=time
        )

        elasticity_coef = (upper - lower) / (2 * displacement * old)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(variables=concs, time=time)
        elasticities[var] = elasticity_coef

    return pd.DataFrame(data=elasticities)


def parameter_elasticities(
    model: Model,
    parameters: list[str] | None = None,
    concs: dict[str, float] | None = None,
    time: float = 0,
    *,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate parameter elasticity coefficients.

    Examples:
        >>> parameter_elasticities(model)
        Rxn    k1    k2
         v1   1.0   0.0
         v2   0.0   1.0
         v3   0.0   0.0

    Args:
        model: Metabolic model instance
        parameters: List of parameters to analyze. Uses all if None
        concs: Initial concentrations {metabolite: value}. Uses model defaults if None
        time: Time point for evaluation
        normalized: Whether to normalize coefficients
        displacement: Relative perturbation size

    Returns:
        DataFrame with parameter elasticities (reactions x parameters)

    """
    concs = model.get_initial_conditions() if concs is None else concs
    parameters = model.get_parameter_names() if parameters is None else parameters

    elasticities = {}

    concs = model.get_initial_conditions() if concs is None else concs
    for par in parameters:
        old = model.parameters[par]

        model.update_parameters({par: old * (1 + displacement)})
        upper = model.get_fluxes(variables=concs, time=time)

        model.update_parameters({par: old * (1 - displacement)})
        lower = model.get_fluxes(variables=concs, time=time)

        # Reset
        model.update_parameters({par: old})
        elasticity_coef = (upper - lower) / (2 * displacement * old)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(variables=concs, time=time)
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
    displacement: float = 1e-4,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol],
) -> tuple[pd.Series, pd.Series]:
    """Calculate response coefficients for a single parameter.

    Internal helper function that computes concentration and flux response
    coefficients using finite differences. The function:
    1. Perturbs the parameter up and down by a small displacement
    2. Calculates steady states for each perturbation
    3. Computes response coefficients from the differences
    4. Optionally normalizes the results

    Args:
        parameter: Name of the parameter to analyze
        model: Metabolic model instance
        y0: Initial conditions as a dictionary {species: value}
        normalized: Whether to normalize the coefficients
        rel_norm: Whether to use relative normalization
        displacement: Relative perturbation size (default: 1e-4)
        integrator: Integrator function to use for steady state calculation

    Returns:
        tuple[pd.Series, pd.Series]: Tuple containing:
            - Series of concentration response coefficients
            - Series of flux response coefficients

    """
    old = model.parameters[parameter]

    model.update_parameters({parameter: old * (1 + displacement)})
    upper = _steady_state_worker(
        model,
        y0=y0,
        rel_norm=rel_norm,
        integrator=integrator,
    )

    model.update_parameters({parameter: old * (1 - displacement)})
    lower = _steady_state_worker(
        model,
        y0=y0,
        rel_norm=rel_norm,
        integrator=integrator,
    )

    conc_resp = (upper.variables - lower.variables) / (2 * displacement * old)
    flux_resp = (upper.fluxes - lower.fluxes) / (2 * displacement * old)
    # Reset
    model.update_parameters({parameter: old})
    if normalized:
        norm = _steady_state_worker(
            model,
            y0=y0,
            rel_norm=rel_norm,
            integrator=integrator,
        )
        conc_resp *= old / norm.variables
        flux_resp *= old / norm.fluxes
    return conc_resp, flux_resp


def response_coefficients(
    model: Model,
    parameters: list[str] | None = None,
    *,
    y0: dict[str, float] | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    rel_norm: bool = False,
    integrator: Callable[[Callable, ArrayLike], IntegratorProtocol] = DefaultIntegrator,
) -> ResponseCoefficients:
    """Calculate response coefficients.

    Examples:
        >>> response_coefficients(model, parameters=["k1", "k2"]).variables
        p    x1    x2
        k1  1.4  1.31
        k2 -1.0 -2.49

    Args:
        model: Metabolic model instance
        parameters: Parameters to analyze. Uses all if None
        y0: Initial conditions as a dictionary {species: value}
        normalized: Whether to normalize coefficients
        displacement: Relative perturbation size
        disable_tqdm: Disable progress bar
        parallel: Whether to parallelize the computation
        max_workers: Maximum number of workers
        rel_norm: Whether to use relative normalization
        integrator: Integrator function to use for steady state calculation

    Returns:
        ResponseCoefficients object containing:
        - Flux response coefficients
        - Concentration response coefficients

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
            integrator=integrator,
        ),
        inputs=list(zip(parameters, parameters, strict=True)),
        cache=None,
        disable_tqdm=disable_tqdm,
        parallel=parallel,
        max_workers=max_workers,
    )
    return ResponseCoefficients(
        variables=pd.DataFrame({k: v[0] for k, v in res.items()}),
        fluxes=pd.DataFrame({k: v[1] for k, v in res.items()}),
    )
