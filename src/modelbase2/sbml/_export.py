from __future__ import annotations

import ast
import inspect
import re
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

import dill
import libsbml

from modelbase2.model import Model
from modelbase2.sbml._data import AtomicUnit, Compartment
from modelbase2.types import Any

__all__ = [
    "RE_LAMBDA_ALGEBRAIC_MODULE_FUNC",
    "RE_LAMBDA_FUNC",
    "RE_LAMBDA_RATE_FUNC",
    "RE_TO_SBML",
    "SBML_DOT",
    "to_sbml",
]

RE_LAMBDA_FUNC = re.compile(r".*(lambda)(.+?):(.*?)")
RE_LAMBDA_RATE_FUNC = re.compile(r".*(lambda)(.+?):(.*?),")
RE_LAMBDA_ALGEBRAIC_MODULE_FUNC = re.compile(r".*(lambda)(.+?):(.*[\(\[].+[\)\]]),")
RE_TO_SBML = re.compile(r"([^0-9_a-zA-Z])")

SBML_DOT = "__SBML_DOT__"


def _handle_return(node: ast.Return) -> ast.expr:
    if (value := node.value) is None:
        msg = "Function without return"
        raise ValueError(msg)

    return value


def _handle_binop(node: ast.BinOp, argmap: dict[str, str]) -> str:
    left = _handle_node(node.left, argmap)
    right = _handle_node(node.right, argmap)

    match node.op:
        case ast.Mult():
            op = "*"
        case ast.Add():
            op = "+"
        case ast.Sub():
            op = "-"
        case ast.Div():
            op = "/"
        case ast.Pow():
            op = "^"
        case _:
            raise NotImplementedError(type(node.op))
    return f"{left} {op} {right}"


def _handle_name(node: ast.Name, argmap: dict[str, str]) -> str:
    return argmap[node.id]


def _handle_assign(node: ast.Assign, argmap: dict[str, str]) -> str:
    value = _handle_node(node.value, argmap)
    target = cast(ast.Name, node.targets[0]).id
    argmap[target] = value
    return value


def _handle_node(node: ast.stmt | ast.expr, argmap: dict[str, str]) -> str:
    if isinstance(node, ast.Return):
        return _handle_node(_handle_return(node), argmap)
    if isinstance(node, ast.BinOp):
        return _handle_binop(node, argmap)
    if isinstance(node, ast.Name):
        return _handle_name(node, argmap)
    if isinstance(node, ast.Assign):
        return _handle_assign(node, argmap)

    raise NotImplementedError(type(node))


def _handle_fn_def(node: ast.FunctionDef, user_args: list[str]) -> str:
    fn_args = [i.arg for i in node.args.args]
    argmap = dict(zip(fn_args, user_args, strict=True))

    code = ""
    for stmt in node.body:
        code = _handle_node(stmt, argmap)
    return code


def _sbmlify_fn(fn: Callable, user_args: list[str]) -> str:
    try:
        source = inspect.getsource(fn)
    except:  # noqa: E722
        source = dill.source.getsource(fn)

    tree = ast.parse(textwrap.dedent(source))
    if not isinstance(fn_def := tree.body[0], ast.FunctionDef):
        msg = "Not a function"
        raise TypeError(msg)
    return _handle_fn_def(fn_def, user_args=user_args)


##########################################################################
# SBML functions
##########################################################################


def _escape_non_alphanumeric(re_sub: Any) -> str:
    """Convert a non-alphanumeric charactor to a string representation of its ascii number."""
    return f"__{ord(re_sub.group(0))}__"


def _convert_id_to_sbml(id_: str, prefix: str) -> str:
    """Add prefix if id startswith number."""
    new_id = RE_TO_SBML.sub(_escape_non_alphanumeric, id_).replace(".", SBML_DOT)
    if not new_id[0].isalpha():
        return f"{prefix}_{new_id}"
    return new_id


def _create_sbml_document() -> libsbml.SBMLDocument:
    """Create an sbml document, into which sbml information can be written.

    Returns:
        doc : libsbml.Document

    """
    # SBML namespaces
    sbml_ns = libsbml.SBMLNamespaces(3, 2)
    sbml_ns.addPackageNamespace("fbc", 2)
    # SBML document
    doc = libsbml.SBMLDocument(sbml_ns)
    doc.setPackageRequired("fbc", flag=False)
    doc.setSBOTerm("SBO:0000004")
    return doc


def _create_sbml_model(
    *,
    model_name: str,
    doc: libsbml.SBMLDocument,
    extent_units: str,
    substance_units: str,
    time_units: str,
) -> libsbml.Model:
    """Create an sbml model.

    Args:
        model_name: Name of the model.
        doc: libsbml.Document
        extent_units: Units for the extent of reactions.
        substance_units: Units for the amount of substances.
        time_units: Units for time.

    Returns:
        sbml_model : libsbml.Model

    """
    name = f"{model_name}_{datetime.now(UTC).date().strftime('%Y-%m-%d')}"
    sbml_model = doc.createModel()
    sbml_model.setId(_convert_id_to_sbml(id_=name, prefix="MODEL"))
    sbml_model.setName(_convert_id_to_sbml(id_=name, prefix="MODEL"))
    sbml_model.setTimeUnits(time_units)
    sbml_model.setExtentUnits(extent_units)
    sbml_model.setSubstanceUnits(substance_units)
    sbml_model_fbc = sbml_model.getPlugin("fbc")
    sbml_model_fbc.setStrict(True)
    return sbml_model


def _create_sbml_units(
    *,
    units: dict[str, AtomicUnit],
    sbml_model: libsbml.Model,
) -> None:
    """Create sbml units out of the meta_info.

    Args:
        units: Dictionary of units to use in the SBML file.
        sbml_model : libsbml Model

    """
    for unit_id, unit in units.items():
        sbml_definition = sbml_model.createUnitDefinition()
        sbml_definition.setId(unit_id)
        sbml_unit = sbml_definition.createUnit()
        sbml_unit.setKind(unit.kind)
        sbml_unit.setExponent(unit.exponent)
        sbml_unit.setScale(unit.scale)
        sbml_unit.setMultiplier(unit.multiplier)


def _create_sbml_compartments(
    *,
    compartments: dict[str, Compartment],
    sbml_model: libsbml.Model,
) -> None:
    for compartment_id, compartment in compartments.items():
        sbml_compartment = sbml_model.createCompartment()
        sbml_compartment.setId(compartment_id)
        sbml_compartment.setName(compartment.name)
        sbml_compartment.setConstant(compartment.is_constant)
        sbml_compartment.setSize(compartment.size)
        sbml_compartment.setSpatialDimensions(compartment.dimensions)
        sbml_compartment.setUnits(compartment.units)


def _create_sbml_variables(
    *,
    model: Model,
    sbml_model: libsbml.Model,
) -> None:
    """Create the variables for the sbml model.

    Args:
        model: Model instance to export.
        sbml_model : libsbml.Model

    """
    for variable_id in model.get_variable_names():
        cpd = sbml_model.createSpecies()
        cpd.setId(_convert_id_to_sbml(id_=variable_id, prefix="CPD"))

        cpd.setConstant(False)
        cpd.setBoundaryCondition(False)
        cpd.setHasOnlySubstanceUnits(False)


def _create_sbml_derived_variables(*, model: Model, sbml_model: libsbml.Model) -> None:
    for name, dv in model.derived_variables.items():
        sbml_ar = sbml_model.createAssignmentRule()
        sbml_ar.setId(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setName(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setVariable(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setConstant(False)
        sbml_ar.setMath(libsbml.parseL3Formula(_sbmlify_fn(dv.fn, dv.args)))


def _create_sbml_parameters(
    *,
    model: Model,
    sbml_model: libsbml.Model,
) -> None:
    """Create the parameters for the sbml model.

    Args:
        model: Model instance to export.
        sbml_model : libsbml.Model

    """
    for parameter_id, value in model.parameters.items():
        k = sbml_model.createParameter()
        k.setId(_convert_id_to_sbml(id_=parameter_id, prefix="PAR"))
        k.setConstant(True)
        k.setValue(float(value))


def _create_sbml_derived_parameters(*, model: Model, sbml_model: libsbml.Model) -> None:
    for name, dp in model.derived_parameters.items():
        sbml_ar = sbml_model.createAssignmentRule()
        sbml_ar.setId(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setName(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setVariable(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setConstant(True)
        sbml_ar.setMath(libsbml.parseL3Formula(_sbmlify_fn(dp.fn, dp.args)))


def _create_sbml_reactions(
    *,
    model: Model,
    sbml_model: libsbml.Model,
) -> None:
    """Create the reactions for the sbml model."""
    for name, rxn in model.reactions.items():
        sbml_rxn = sbml_model.createReaction()
        sbml_rxn.setId(_convert_id_to_sbml(id_=name, prefix="RXN"))
        sbml_rxn.setName(name)
        sbml_rxn.setFast(False)
        # sbml_rxn.setReversible(model.rates[name]["reversible"])

        for compound_id, factor in rxn.stoichiometry.items():
            if isinstance(factor, float):
                sref = (
                    sbml_rxn.createReactant()
                    if factor < 0
                    else sbml_rxn.createProduct()
                )
                sref.setSpecies(_convert_id_to_sbml(id_=compound_id, prefix="CPD"))
                sref.setStoichiometry(abs(factor))
                sref.setConstant(False)

        for compound_id in rxn.get_modifiers(model):
            sref = sbml_rxn.createModifier()
            sref.setSpecies(_convert_id_to_sbml(id_=compound_id, prefix="CPD"))

        sbml_rxn.createKineticLaw().setMath(
            libsbml.parseL3Formula(_sbmlify_fn(rxn.fn, rxn.args))
        )


def _model_to_sbml(
    model: Model,
    *,
    model_name: str,
    units: dict[str, AtomicUnit],
    extent_units: str,
    substance_units: str,
    time_units: str,
    compartments: dict[str, Compartment],
) -> libsbml.SBMLDocument:
    """Export model to sbml."""
    doc = _create_sbml_document()
    sbml_model = _create_sbml_model(
        model_name=model_name,
        doc=doc,
        extent_units=extent_units,
        substance_units=substance_units,
        time_units=time_units,
    )
    _create_sbml_units(units=units, sbml_model=sbml_model)
    _create_sbml_compartments(compartments=compartments, sbml_model=sbml_model)
    # Actual model components
    _create_sbml_parameters(model=model, sbml_model=sbml_model)
    _create_sbml_derived_parameters(model=model, sbml_model=sbml_model)
    _create_sbml_variables(model=model, sbml_model=sbml_model)
    _create_sbml_derived_variables(model=model, sbml_model=sbml_model)
    _create_sbml_reactions(model=model, sbml_model=sbml_model)
    return doc


def _default_compartments(
    compartments: dict[str, Compartment] | None,
) -> dict[str, Compartment]:
    if compartments is None:
        return {
            "c": Compartment(
                name="cytosol",
                dimensions=3,
                size=1,
                units="litre",
                is_constant=True,
            )
        }
    return compartments


def _default_units(units: dict[str, AtomicUnit] | None) -> dict[str, AtomicUnit]:
    if units is None:
        return {
            "per_second": AtomicUnit(
                kind=libsbml.UNIT_KIND_SECOND,
                exponent=-1,
                scale=0,
                multiplier=1,
            )
        }
    return units


def to_sbml(
    model: Model,
    filename: str,
    model_name: str,
    *,
    units: dict[str, AtomicUnit] | None = None,
    compartments: dict[str, Compartment] | None = None,
    extent_units: str = "mole",
    substance_units: str = "mole",
    time_units: str = "second",
) -> str | None:
    """Export a metabolic model to an SBML file.

    Args:
        model: Model instance to export.
        filename: Name of the SBML file to create.
        model_name: Name of the model.
        units: Dictionary of units to use in the SBML file (default: None).
        compartments: Dictionary of compartments to use in the SBML file (default: None).
        extent_units: Units for the extent of reactions (default: "mole").
        substance_units: Units for the amount of substances (default: "mole").
        time_units: Units for time (default: "second").

    Returns:
        str | None: None if the export is successful.

    """
    doc = _model_to_sbml(
        model=model,
        model_name=model_name,
        units=_default_units(units),
        extent_units=extent_units,
        substance_units=substance_units,
        time_units=time_units,
        compartments=_default_compartments(compartments),
    )

    libsbml.writeSBMLToFile(doc, filename)
    return None
