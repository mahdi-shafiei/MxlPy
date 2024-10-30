from __future__ import annotations

__all__ = [
    "ParameterMixin",
]

import warnings
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any, Self

from .basemodel import BaseModel
from .utils import (
    convert_id_to_sbml,
    get_formatted_function_source_code,
    warning_on_one_line,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import libsbml

warnings.formatwarning = warning_on_one_line  # type: ignore


@dataclass
class DerivedParameter:
    function: Callable
    parameters: list[str]

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]


class ParameterMixin(BaseModel):
    """Adding parameter functions."""

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        self.derived_parameters: dict[str, DerivedParameter] = {}
        self._parameters: dict[str, float] = {}
        self._derived_parameter_order: list[str] = []

        # Bookkeeping
        self._derived_from_parameters: set[str] = set()

        if parameters is not None:
            self.add_parameters(parameters=parameters)
        self.initialization_parameters = self._parameters.copy()

    @property
    def parameters(self) -> dict[str, float]:
        return self.get_all_parameters()

    ##########################################################################
    # Parameter functions
    ##########################################################################

    def add_parameter(
        self,
        parameter_name: str,
        parameter_value: float,
        *,
        update_derived: bool = True,  # noqa: ARG002
    ) -> Self:
        """Add a new parameter to the model."""
        self._check_and_insert_ids([parameter_name], context="add_parameter")
        self._parameters[parameter_name] = parameter_value
        return self

    def add_parameters(self, parameters: dict[str, float]) -> Self:
        """Add multiple parameters to the model"""
        for parameter_name, parameter_value in parameters.items():
            self.add_parameter(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
            )
        return self

    def update_parameter(
        self,
        parameter_name: str,
        parameter_value: float,
        *,
        update_derived: bool = True,  # noqa: ARG002
    ) -> Self:
        """Update an existing model parameter.

        Warns:
        -----
        UserWarning
            If parameter is not in the model

        """
        if parameter_name not in self._parameters:
            if parameter_name in self._derived_from_parameters:
                msg = f"{parameter_name} is a derived parameter"
                raise ValueError(msg)
            warnings.warn(
                f"Key {parameter_name} is not in the model. Adding.",
                stacklevel=1,
            )
        self._parameters[parameter_name] = parameter_value
        return self

    def scale_parameter(
        self,
        parameter_name: str,
        factor: float,
        *,
        verbose: bool = False,
    ) -> Self:
        self.update_parameter(
            parameter_name,
            self._parameters[parameter_name] * factor,
        )
        if verbose:
            pass
        return self

    def update_parameters(
        self,
        parameters: dict[str, float],
    ) -> Self:
        """Update multiple existing model parameters.

        See Also
        --------
        update_parameter

        """
        for parameter_name, parameter_value in parameters.items():
            self.update_parameter(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
            )
        return self

    def add_and_update_parameter(
        self,
        parameter_name: str,
        parameter_value: float,
        *,
        update_derived: bool = True,
    ) -> Self:
        """Add a new or update an existing parameter."""
        if parameter_name not in self._ids:
            self.add_parameter(
                parameter_name,
                parameter_value,
                update_derived=update_derived,
            )
        else:
            self.update_parameter(
                parameter_name,
                parameter_value,
                update_derived=update_derived,
            )
        return self

    def add_and_update_parameters(
        self,
        parameters: dict[str, float],
    ) -> Self:
        """Add new and update existing model parameters.

        See Also
        --------
        add_and_update_parameter

        """
        for parameter_name, parameter_value in parameters.items():
            self.add_and_update_parameter(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
            )
        return self

    def remove_parameter(self, parameter_name: str) -> Self:
        """Remove a parameter from the model."""
        del self._parameters[parameter_name]
        self._remove_ids([parameter_name])
        return self

    def remove_parameters(self, parameter_names: Iterable[str]) -> Self:
        """Remove multiple parameters from the model.

        See Also
        --------
        remove_parameter

        """
        for parameter_name in parameter_names:
            self.remove_parameter(parameter_name=parameter_name)
        return self

    ##########################################################################
    # Derived parameter functions
    ##########################################################################

    def _sort_derived_parameters(self, max_iterations: int = 10_000) -> None:
        available = set(self.get_parameter_names())
        par_order = []
        pars_to_sort: SimpleQueue = SimpleQueue()
        for k, v in self.derived_parameters.items():
            pars_to_sort.put((k, set(v.parameters)))

        last_name = None
        i = 0
        name: str
        args: set[str]
        while True:
            try:
                name, args = pars_to_sort.get_nowait()
            except Empty:
                break
            if args.issubset(available):
                available.add(name)
                par_order.append(name)
            else:
                if last_name == name:
                    par_order.append(name)
                    break
                pars_to_sort.put((name, args))
                last_name = name
            i += 1
            if i > max_iterations:
                to_sort = []
                while True:
                    try:
                        to_sort.append(pars_to_sort.get_nowait()[0])
                    except Empty:
                        break

                msg = (
                    "Exceeded max iterations on derived parameter module sorting. "
                    "Check if there are circular references.\n"
                    f"Available: {to_sort}\n"
                    f"Order: {par_order}"
                )
                raise ValueError(msg)
        self._derived_parameter_order = par_order

    def add_derived_parameter(
        self,
        parameter_name: str,
        function: Callable,
        parameters: list[str],
    ) -> Self:
        """Add a derived parameter.

        Derived parameters are calculated from other model parameters and dynamically updated
        on any changes.
        """
        self.derived_parameters[parameter_name] = DerivedParameter(
            function=function,
            parameters=parameters,
        )
        for parameter in parameters:
            self._derived_from_parameters.add(parameter)
        self._check_and_insert_ids([parameter_name], context="add_derived_parameter")
        self._sort_derived_parameters()
        return self

    def update_derived_parameter(
        self,
        parameter_name: str,
        function: Callable | None,
        parameters: list[str] | None,
    ) -> Self:
        old = self.derived_parameters[parameter_name]
        if function is None:
            function = old.function
        if parameters is None:
            parameters = old.parameters

        self.derived_parameters[parameter_name].function = function
        self.derived_parameters[parameter_name].parameters = parameters
        self._sort_derived_parameters()
        return self

    def remove_derived_parameter(self, parameter_name: str) -> Self:
        """Remove a derived parameter from the model."""
        old_parameter = self.derived_parameters.pop(parameter_name)
        derived_from = old_parameter["parameters"]
        for i in derived_from:
            if all(i not in j["parameters"] for j in self.derived_parameters.values()):
                self._derived_from_parameters.remove(i)
        self._sort_derived_parameters()
        self._remove_ids([parameter_name])
        return self

    def restore_initialization_parameters(self) -> Self:
        """Restore parameters to initialization parameters."""
        self._parameters = self.initialization_parameters.copy()
        return self

    def get_parameter(self, parameter_name: str) -> float:
        """Return the value of a single parameter."""
        if parameter_name in self.derived_parameters:
            msg = (
                f"Parameter {parameter_name} is a derived parameter. "
                "Use `.parameters[parameter_name]` "
                "or `get_derived_parameter_value(parameter_name)`"
            )
            raise ValueError(msg)
        return self._parameters[parameter_name]

    def get_derived_parameter(self, parameter_name: str) -> DerivedParameter:
        return self.derived_parameters[parameter_name]

    def get_derived_parameter_value(self, parameter_name: str) -> float:
        if parameter_name in self._parameters:
            msg = (
                f"Parameter {parameter_name} is a normal parameter. "
                "Use `.parameters[parameter_name]` "
                "or `.get_parameter(parameter_name)`"
            )
            raise ValueError(msg)

        return self.parameters[parameter_name]

    def get_parameters(self) -> dict[str, float]:
        """Return all parameters."""
        return dict(self._parameters)

    def get_all_parameters(self) -> dict[str, float]:
        all_params = self._parameters.copy()

        for parameter_name in self._derived_parameter_order:
            derived_parameter = self.derived_parameters[parameter_name]
            all_params[parameter_name] = derived_parameter.function(
                *(all_params[i] for i in derived_parameter.parameters)
            )

        return all_params

    def get_parameter_names(self) -> list[str]:
        """Return names of all parameters"""
        return list(self._parameters)

    def get_all_parameter_names(self) -> set[str]:
        """Return names of all parameters"""
        return set(self._parameters) | set(self.derived_parameters)

    ##########################################################################
    # Source code functions
    ##########################################################################
    def _generate_constant_parameters_source_code(self) -> str:
        """Generate modelbase source code for parameters.

        This is mainly used for the generate_model_source_code function.

        Returns
        -------
        parameter_modelbase_code : str
            Source code generating the modelbase parameters

        """
        parameters = repr(dict(self._parameters.items()))
        return f"m.add_parameters(parameters={parameters})"

    def _generate_derived_parameters_source_code(self) -> tuple[str, str]:
        """Generate modelbase source code for parameters.

        This is mainly used for the generate_model_source_code function.

        Returns
        -------
        parameter_modelbase_code : str
            Source code generating the modelbase parameters

        """
        fns: set[str] = set()
        pars: list[str] = []
        for name, module in self.derived_parameters.items():
            function = module["function"]
            parameters = module["parameters"]

            function_code = get_formatted_function_source_code(
                function_name=name, function=function, function_type="module"
            )
            fns.add(function_code)
            pars.append(
                "m.add_derived_parameter(\n"
                f"    parameter_name={name!r},\n"
                f"    function={function.__name__},\n"
                f"    parameters={parameters},\n"
                ")"
            )
        return "\n".join(sorted(fns)), "\n".join(pars)

    def _generate_parameters_source_code(self) -> tuple[str, str, str]:
        return (
            self._generate_constant_parameters_source_code(),
            *self._generate_derived_parameters_source_code(),
        )

    ##########################################################################
    # SBML functions
    ##########################################################################

    def _create_sbml_parameters(self, *, sbml_model: libsbml.Model) -> None:
        """Create the parameters for the sbml model.

        Parameters
        ----------
        sbml_model : libsbml.Model

        """
        for parameter_id, value in self._parameters.items():
            k = sbml_model.createParameter()
            k.setId(convert_id_to_sbml(id_=parameter_id, prefix="PAR"))
            k.setConstant(True)
            k.setValue(float(value))
