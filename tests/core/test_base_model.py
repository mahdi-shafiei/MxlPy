# type: ignore

from __future__ import annotations

import pathlib
import tempfile
import unittest
from datetime import datetime

import libsbml

from modelbase2.ode import Model
from modelbase2.sbml import convert_sbml_id


class BaseModelTests(unittest.TestCase):
    def test_init_empty(self):
        model = Model()
        self.assertTrue(model)
        self.assertEqual(
            model.meta_info["model"].__dict__,
            {
                "sbo": "SBO:0000062",
                "id": f"modelbase-model-{datetime.now().date().strftime('%Y-%m-%d')}",
                "name": "modelbase-model",
                "units": {
                    "per_second": {
                        "kind": 28,
                        "exponent": -1,
                        "scale": 0,
                        "multiplier": 1,
                    }
                },
                "compartments": {
                    "c": {
                        "name": "cytosol",
                        "is_constant": True,
                        "size": 1,
                        "spatial_dimensions": 3,
                        "units": "litre",
                    }
                },
                "notes": {},
            },
        )

    def test_init_meta_info(self):
        model = Model(meta_info={"name": "my-model"})
        self.assertEqual(model.meta_info["model"].name, "my-model")

    def test_enter(self):
        model = Model()
        with model as m_dup:
            self.assertIsNot(model, m_dup)

    def test_exit(self):
        model = Model()
        with model:
            model.test = 1
        with self.assertRaises(AttributeError):
            model.test

    def test_copy(self):
        model = Model()
        m_copy = model.copy()
        self.assertIsNot(model, m_copy)

    def test_add_meta_info(self):
        model = Model()
        model.update_meta_info(component="model", meta_info={"sbo": "123"})
        self.assertEqual(
            model.meta_info["model"].__dict__,
            {
                "sbo": "123",
                "id": f"modelbase-model-{datetime.now().date().strftime('%Y-%m-%d')}",
                "name": "modelbase-model",
                "units": {
                    "per_second": {
                        "kind": 28,
                        "exponent": -1,
                        "scale": 0,
                        "multiplier": 1,
                    }
                },
                "compartments": {
                    "c": {
                        "name": "cytosol",
                        "is_constant": True,
                        "size": 1,
                        "spatial_dimensions": 3,
                        "units": "litre",
                    }
                },
                "notes": {},
            },
        )


class SBMLTests(unittest.TestCase):
    def test_create_sbml_document(self):
        model = Model()
        doc = model._create_sbml_document()
        self.assertEqual(doc.getLevel(), 3)
        self.assertEqual(doc.getVersion(), 2)
        self.assertEqual(doc.getSBOTerm(), 62)

    def test_create_sbml_model(self):
        model = Model()
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        self.assertEqual(
            convert_sbml_id(sbml_model.getId(), prefix="MODEL"),
            f"modelbase-model-{datetime.now().date().strftime('%Y-%m-%d')}",
        )
        self.assertEqual(
            convert_sbml_id(sbml_model.getName(), prefix="MODEL"), "modelbase-model"
        )
        self.assertEqual(sbml_model.getTimeUnits(), "second")
        self.assertEqual(sbml_model.getExtentUnits(), "mole")
        self.assertEqual(sbml_model.getSubstanceUnits(), "mole")

    def test_create_sbml_units(self):
        model = Model()
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_units(sbml_model=sbml_model)

        unit_definition = sbml_model.unit_definitions[0]
        self.assertEqual(unit_definition.getId(), "per_second")

        unit = unit_definition.getListOfUnits()[0]
        self.assertEqual(unit.getKind(), libsbml.UNIT_KIND_SECOND)
        self.assertEqual(unit.getExponent(), -1)
        self.assertEqual(unit.getScale(), 0)
        self.assertEqual(unit.getMultiplier(), 1)

    def test_create_sbml_compartments(self):
        model = Model()
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_compartments(sbml_model=sbml_model)

        compartment = sbml_model.getListOfCompartments()[0]
        self.assertEqual(compartment.getId(), "c")
        self.assertEqual(compartment.getName(), "cytosol")
        self.assertEqual(compartment.getConstant(), True)
        self.assertEqual(compartment.getSize(), 1.0)
        self.assertEqual(compartment.getSpatialDimensions(), 3)
        self.assertEqual(compartment.getUnits(), "litre")

    def test_write_sbml_model_file(self):
        model = Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "testfile.xml"
            model.write_sbml_model(str(path))
            self.assertTrue(path.is_file())

    def test_write_sbml_model_string(self):
        model = Model()
        s = model.write_sbml_model()
        self.assertTrue(isinstance(s, str))
