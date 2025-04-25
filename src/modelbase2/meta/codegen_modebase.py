"""Generate modelbase code from a model."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from modelbase2.meta.source_tools import get_fn_source
from modelbase2.types import Derived

__all__ = ["generate_modelbase_code"]

if TYPE_CHECKING:
    from modelbase2.model import Model


def generate_modelbase_code(model: Model) -> str:
    """Generate a modelbase model from a model."""
    functions = {}

    # Variables and parameters
    variables = model.variables
    parameters = model.parameters

    # Derived
    derived_source = []
    for k, rxn in model.derived.items():
        fn = rxn.fn
        fn_name = fn.__name__
        functions[fn_name] = get_fn_source(fn)

        derived_source.append(
            f"""        .add_derived(
                "{k}",
                fn={fn_name},
                args={rxn.args},
            )"""
        )

    # Reactions
    reactions_source = []
    for k, rxn in model.reactions.items():
        fn = rxn.fn
        fn_name = fn.__name__
        functions[fn_name] = get_fn_source(fn)
        stoichiometry: list[str] = []
        for var, stoich in rxn.stoichiometry.items():
            if isinstance(stoich, Derived):
                functions[fn_name] = get_fn_source(fn)
                args = ", ".join(f'"{k}"' for k in stoich.args)
                stoich = (  # noqa: PLW2901
                    f"""Derived(name="{var}", fn={fn.__name__}, args=[{args}])"""
                )
            stoichiometry.append(f""""{var}": {stoich}""")

        reactions_source.append(
            f"""        .add_reaction(
                "{k}",
                fn={fn_name},
                args={rxn.args},
                stoichiometry={{{",".join(stoichiometry)}}},
            )"""
        )

    # Surrogates
    if len(model._surrogates) > 0:  # noqa: SLF001
        warnings.warn(
            "Generating code for Surrogates not yet supported.",
            stacklevel=1,
        )

    # Combine all the sources
    functions_source = "\n".join(functions.values())
    source = [
        "from modelbase2 import Model\n",
        functions_source,
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
    ]
    if len(parameters) > 0:
        source.append(f"        .add_parameters({parameters})")
    if len(variables) > 0:
        source.append(f"        .add_variables({variables})")
    if len(derived_source) > 0:
        source.append("\n".join(derived_source))
    if len(reactions_source) > 0:
        source.append("\n".join(reactions_source))

    source.append("    )")

    return "\n".join(source)
