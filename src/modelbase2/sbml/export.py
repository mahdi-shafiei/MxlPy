import re

import libsbml

from modelbase2.types import Any, ModelProtocol

RE_LAMBDA_FUNC = re.compile(r".*(lambda)(.+?):(.*?)")
RE_LAMBDA_RATE_FUNC = re.compile(r".*(lambda)(.+?):(.*?),")
RE_LAMBDA_ALGEBRAIC_MODULE_FUNC = re.compile(r".*(lambda)(.+?):(.*[\(\[].+[\)\]]),")
RE_TO_SBML = re.compile(r"([^0-9_a-zA-Z])")
RE_FROM_SBML = re.compile(r"__(\d+)__")
SBML_DOT = "__SBML_DOT__"

##########################################################################
# SBML functions
##########################################################################


def escape_non_alphanumeric(re_sub: Any) -> str:
    """Convert a non-alphanumeric charactor to a string representation of its ascii number."""
    return f"__{ord(re_sub.group(0))}__"


def ascii_to_character(re_sub: Any) -> str:
    """Convert an escaped non-alphanumeric character."""
    return chr(int(re_sub.group(1)))


def convert_id_to_sbml(id_: str, prefix: str) -> str:
    """Add prefix if id startswith number."""
    new_id = RE_TO_SBML.sub(escape_non_alphanumeric, id_).replace(".", SBML_DOT)
    if not new_id[0].isalpha():
        return f"{prefix}_{new_id}"
    return new_id


def convert_sbml_id(sbml_id: str, prefix: str) -> str:
    """Convert an model object id to sbml-compatible string.

    Adds a prefix if the id starts with a number.
    """
    new_id = sbml_id.replace(SBML_DOT, ".")
    new_id = RE_FROM_SBML.sub(ascii_to_character, new_id)
    return new_id.lstrip(f"{prefix}_")


def _create_sbml_document(model: ModelProtocol) -> libsbml.SBMLDocument:
    """Create an sbml document, into which sbml information can be written.

    Returns
    -------
    doc : libsbml.Document

    """
    # SBML namespaces
    sbml_ns = libsbml.SBMLNamespaces(3, 2)
    sbml_ns.addPackageNamespace("fbc", 2)
    # SBML document
    doc = libsbml.SBMLDocument(sbml_ns)
    doc.setPackageRequired("fbc", flag=False)
    doc.setSBOTerm(model.meta_info["model"].sbo)
    return doc


def _create_sbml_model(
    model: ModelProtocol, *, doc: libsbml.SBMLDocument
) -> libsbml.Model:
    """Create an sbml model.

    Parameters
    ----------
    doc : libsbml.Document

    Returns
    -------
    sbml_model : libsbml.Model

    """
    sbml_model = doc.createModel()
    sbml_model.setId(
        convert_id_to_sbml(id_=model.meta_info["model"].id, prefix="MODEL")
    )
    sbml_model.setName(
        convert_id_to_sbml(id_=model.meta_info["model"].name, prefix="MODEL")
    )
    sbml_model.setTimeUnits("second")
    sbml_model.setExtentUnits("mole")
    sbml_model.setSubstanceUnits("mole")
    sbml_model_fbc = sbml_model.getPlugin("fbc")
    sbml_model_fbc.setStrict(True)
    return sbml_model


def _create_sbml_units(model: ModelProtocol, *, sbml_model: libsbml.Model) -> None:
    """Create sbml units out of the meta_info.

    Parameters
    ----------
    sbml_model : libsbml.Model

    """
    for unit_id, unit in model.meta_info["model"].units.items():
        sbml_definition = sbml_model.createUnitDefinition()
        sbml_definition.setId(unit_id)
        sbml_unit = sbml_definition.createUnit()
        sbml_unit.setKind(unit["kind"])
        sbml_unit.setExponent(unit["exponent"])
        sbml_unit.setScale(unit["scale"])
        sbml_unit.setMultiplier(unit["multiplier"])


def _create_sbml_compartments(
    model: ModelProtocol, *, sbml_model: libsbml.Model
) -> None:
    """Create the compartments for the sbml model.

    Since modelbase does not enforce any compartments, so far
    only a cytosol placeholder is introduced.

    Parameters
    ----------
    sbml_model : libsbml.Model

    """
    for compartment_id, compartment in model.meta_info["model"].compartments.items():
        sbml_compartment = sbml_model.createCompartment()
        sbml_compartment.setId(compartment_id)
        sbml_compartment.setName(compartment["name"])
        sbml_compartment.setConstant(compartment["is_constant"])
        sbml_compartment.setSize(compartment["size"])
        sbml_compartment.setSpatialDimensions(compartment["spatial_dimensions"])
        sbml_compartment.setUnits(compartment["units"])


def write_sbml_model(model: ModelProtocol, filename: str | None = None) -> str | None:
    """Write the model to an sbml file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    doc : libsbml.Document

    """
    doc = model._model_to_sbml()
    if filename is not None:
        libsbml.writeSBMLToFile(doc, filename)
        return None
    model_str: str = libsbml.writeSBMLToString(doc)
    return model_str


def _create_sbml_variables(
    model: ModelProtocol,
    *,
    sbml_model: libsbml.Model,
) -> None:
    """Create the variables for the sbml model.

    Parameters
    ----------
    sbml_model : libsbml.Model

    """
    for variable_id in model.get_variable_names():
        cpd = sbml_model.createSpecies()
        cpd.setId(convert_id_to_sbml(id_=variable_id, prefix="CPD"))

        cpd.setConstant(False)
        cpd.setBoundaryCondition(False)
        cpd.setHasOnlySubstanceUnits(False)


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


def _create_sbml_rates(self, *, sbml_model: libsbml.Model) -> None:
    """Convert the rates into sbml reactions.

    Parameters
    ----------
    sbml_model : libsbml.Model

    """
    for rate_id, rate in self.rates.items():
        rxn = sbml_model.createReaction()
        rxn.setId(convert_id_to_sbml(id_=rate_id, prefix="RXN"))
        rxn.setFast(False)
        rxn.setReversible(rate.reversible)

        substrates: defaultdict[str, int] = defaultdict(int)
        products: defaultdict[str, int] = defaultdict(int)
        for compound in rate.substrates:
            substrates[compound] += 1
        for compound in rate.products:
            products[compound] += 1

        for compound, stoichiometry in substrates.items():
            sref = rxn.createReactant()
            sref.setSpecies(convert_id_to_sbml(id_=compound, prefix="CPD"))
            sref.setStoichiometry(stoichiometry)
            sref.setConstant(False)

        for compound, stoichiometry in products.items():
            sref = rxn.createProduct()
            sref.setSpecies(convert_id_to_sbml(id_=compound, prefix="CPD"))
            sref.setStoichiometry(stoichiometry)
            sref.setConstant(False)

        for compound in rate.modifiers:
            sref = rxn.createModifier()
            sref.setSpecies(convert_id_to_sbml(id_=compound, prefix="CPD"))


def _create_sbml_stoichiometries(self, *, sbml_model: libsbml.Model) -> None:
    """Create the reactions for the sbml model.

    Parameters
    ----------
    sbml_model : libsbml.Model

    """
    for rate_id, stoichiometry in self.stoichiometries.items():
        rxn = sbml_model.createReaction()
        rxn.setId(convert_id_to_sbml(id_=rate_id, prefix="RXN"))

        for compound_id, factor in stoichiometry.items():
            sref = rxn.createReactant() if factor < 0 else rxn.createProduct()
            sref.setSpecies(convert_id_to_sbml(id_=compound_id, prefix="CPD"))
            sref.setStoichiometry(abs(factor))
            sref.setConstant(True)


def _create_sbml_reactions(self, *, sbml_model: libsbml.Model) -> None:
    """Create the reactions for the sbml model."""
    for rate_id, stoichiometry in self.stoichiometries.items():
        rate = self.meta_info["rates"][rate_id]
        rxn = sbml_model.createReaction()
        rxn.setId(convert_id_to_sbml(id_=rate_id, prefix="RXN"))
        name = rate.common_name
        if name:
            rxn.setName(name)
        rxn.setFast(False)
        rxn.setReversible(self.rates[rate_id]["reversible"])

        for compound_id, factor in stoichiometry.items():
            sref = rxn.createReactant() if factor < 0 else rxn.createProduct()
            sref.setSpecies(convert_id_to_sbml(id_=compound_id, prefix="CPD"))
            sref.setStoichiometry(abs(factor))
            sref.setConstant(False)

        for compound in self.rates[rate_id]["modifiers"]:
            sref = rxn.createModifier()
            sref.setSpecies(convert_id_to_sbml(id_=compound, prefix="CPD"))

        function = rate.sbml_function
        if function is not None:
            kinetic_law = rxn.createKineticLaw()
            kinetic_law.setMath(libsbml.parseL3Formula(function))


def _model_to_sbml(self) -> libsbml.SBMLDocument:
    """Export model to sbml."""
    doc = self._create_sbml_document()
    sbml_model = self._create_sbml_model(doc=doc)
    self._create_sbml_units(sbml_model=sbml_model)
    self._create_sbml_compartments(sbml_model=sbml_model)
    self._create_sbml_compounds(sbml_model=sbml_model)
    if bool(self.algebraic_modules):
        self._create_sbml_algebraic_modules(_sbml_model=sbml_model)
    self._create_sbml_reactions(sbml_model=sbml_model)
    return doc
