# from __future__ import annotations

# __all__ = [
#     "BASE_MODEL_TYPE",
#     "RATE_MODEL_TYPE",
#     "Simulator",
#     "_AbstractRateModel",
#     "_AbstractStoichiometricModel",
#     "_BaseRateSimulator",
#     "_BaseSimulator",
#     "_LabelSimulate",
#     "_LinearLabelSimulate",
#     "_Simulate",
# ]

# import warnings
# from typing import TYPE_CHECKING, overload


# from .abstract_simulator import _BaseRateSimulator, _BaseSimulator
# from .labelsimulator import _LabelSimulate
# from .linearlabelsimulator import _LinearLabelSimulate
# from .simulator import Simulator

# if TYPE_CHECKING:
#     from modelbase2.types import ArrayLike

# try:
#     from modelbase2.integrators import Assimulo

#     default_integrator: type[AbstractIntegrator] = Assimulo
# except ImportError:  # pragma: no cover
#     warnings.warn(
#         "Assimulo not found, disabling sundials support.",
#         stacklevel=1,
#     )
#     default_integrator = Scipy


# @overload
# def Simulator(
#     model: _Model,
#     integrator: type[AbstractIntegrator] = default_integrator,
#     y0: ArrayLike | None = None,
# ) -> _Simulate: ...


# @overload
# def Simulator(
#     model: _LabelModel,
#     integrator: type[AbstractIntegrator] = default_integrator,
#     y0: ArrayLike | None = None,
# ) -> _LabelSimulate: ...


# @overload
# def Simulator(
#     model: _LinearLabelModel,
#     integrator: type[AbstractIntegrator] = default_integrator,
#     y0: ArrayLike | None = None,
# ) -> _LinearLabelSimulate: ...


# def Simulator(  # noqa: N802
#     model: _LabelModel | _LinearLabelModel | _Model,
#     integrator: type[AbstractIntegrator] = default_integrator,
#     y0: ArrayLike | None = None,
# ) -> _LabelSimulate | _LinearLabelSimulate | _Simulate:
#     """Choose the simulator class according to the model type.

#     If a simulator different than assimulo is required, it can be chosen
#     by the integrator argument.

#     Parameters
#     ----------
#     model : modelbase.model
#         The model instance

#     Returns
#     -------
#     Simulate : object
#         A simulate object according to the model type

#     """
#     if isinstance(model, _LabelModel):
#         return _LabelSimulate(
#             model=model,
#             integrator=integrator,
#             y0=y0,
#         )
#     if isinstance(model, _LinearLabelModel):
#         return _LinearLabelSimulate(
#             model=model,
#             integrator=integrator,
#             y0=y0,
#         )
#     if isinstance(model, _Model):
#         return _Simulate(
#             model=model,
#             integrator=integrator,
#             y0=y0,
#         )
#     raise NotImplementedError
