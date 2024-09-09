# type: ignore

from __future__ import annotations

import unittest

from modelbase2.ode import Model


class ModelErrorTests(unittest.TestCase):
    def test_add_compound_error_non_string(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound=tuple())
        with self.assertRaises(TypeError):
            model.add_compound(compound=list())
        with self.assertRaises(TypeError):
            model.add_compound(compound=dict())
        with self.assertRaises(TypeError):
            model.add_compound(compound=set())
        with self.assertRaises(TypeError):
            model.add_compound(compound=int())
        with self.assertRaises(TypeError):
            model.add_compound(compound=float())
        with self.assertRaises(TypeError):
            model.add_compound(compound=None)

    def test_add_compound_error_time(self):
        model = Model()
        with self.assertRaises(KeyError):
            model.add_compound(compound="time")

    def test_add_compound_error_duplicate(self):
        model = Model()
        model.add_compound(compound="x")
        with self.assertWarns(UserWarning):
            model.add_compound(compound="x")
            self.assertEqual(model.compounds, ["x"])

    def test_add_compound_duplicate(self):
        model = Model()
        model.add_compound(compound="x")
        with self.assertWarns(UserWarning):
            model.add_compound(compound="x")
        self.assertEqual(model.compounds, ["x"])

    def test_add_compound_duplicate_update_nometa(self):
        model = Model()
        model.add_compound(compound="x", **{"common_name": "A"})
        with self.assertWarns(UserWarning):
            model.add_compound(compound="x")
            self.assertEqual(model.meta_info["compounds"]["x"].common_name, None)

    def test_add_compound_duplicate_update_newmeta(self):
        model = Model()
        model.add_compound(compound="x", **{"common_name": "A"})
        with self.assertWarns(UserWarning):
            model.add_compound(compound="x", **{"common_name": "X"})
            self.assertEqual(model.meta_info["compounds"]["x"].common_name, "X")


class ModelTests(unittest.TestCase):
    """Tests for compound methods"""

    ############################################################################
    # Adding compounds
    # This should be type checked, we really only want to have compounds as
    # strings, not as everything else
    ############################################################################

    def test_add_compound_str(self):
        model = Model()
        model.add_compound(compound="x")
        self.assertEqual(model.compounds, ["x"])

    def test_add_compound_fail_on_int(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound=1)

    def test_add_compound_fail_on_float(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound=1.0)

    def test_add_compound_fail_on_list(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound=["a"])

    def test_add_compound_fail_on_tuple(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound=("a",))

    def test_add_compound_fail_on_set(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound={"a"})

    def test_add_compound_fail_on_dict(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compound(compound={"a": 1})

    def test_add_compounds_str(self):
        model = Model()
        model.add_compounds(compounds="xyz")
        self.assertEqual(model.compounds, ["x", "y", "z"])

    def test_add_compounds_tuple(self):
        model = Model()
        model.add_compounds(compounds=("x", "y", "z"))
        self.assertEqual(model.compounds, ["x", "y", "z"])

    def test_add_compounds_list(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        self.assertEqual(model.compounds, ["x", "y", "z"])

    def test_add_compounds_set(self):
        model = Model()
        model.add_compounds(compounds={"x", "y", "z"})
        self.assertEqual(set(model.compounds), set(["x", "y", "z"]))

    def test_add_compounds_dict(self):
        model = Model()
        model.add_compounds(
            compounds=("x", "y", "z"),
            meta_info={
                "x": {"common_name": "cpd-x"},
                "y": {"common_name": "cpd-y"},
                "z": {"common_name": "cpd-z"},
            },
        )
        self.assertEqual(model.compounds, ["x", "y", "z"])
        self.assertEqual(model.meta_info["compounds"]["x"].common_name, "cpd-x")
        self.assertEqual(model.meta_info["compounds"]["y"].common_name, "cpd-y")
        self.assertEqual(model.meta_info["compounds"]["z"].common_name, "cpd-z")

    def test_add_compounds_fail_on_int(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compounds(compounds=1)

    def test_add_compounds_fail_on_float(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compounds(compounds=1.0)

    def test_add_compounds_fail_on_none(self):
        model = Model()
        with self.assertRaises(TypeError):
            model.add_compounds(compounds=None)

    def test_add_compounds_duplicate_sets(self):
        model = Model()
        model.add_compounds(compounds={"x", "y", "z"})
        with self.assertWarns(UserWarning):
            model.add_compounds(compounds={"x", "y", "z"})
            self.assertEqual(set(model.compounds), set(["x", "y", "z"]))

    ############################################################################
    # Removing compounds
    ############################################################################

    def test_remove_compound_beginning(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compound(compound="x")
        self.assertEqual(model.compounds, ["y", "z"])

    def test_remove_compound_middle(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compound(compound="y")
        self.assertEqual(model.compounds, ["x", "z"])

    def test_remove_compound_end(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compound(compound="z")
        self.assertEqual(model.compounds, ["x", "y"])

    def test_remove_compounds_str(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compounds(compounds="xy")
        self.assertEqual(model.compounds, ["z"])

    def test_remove_compounds_tuple(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compounds(compounds=("x", "y"))
        self.assertEqual(model.compounds, ["z"])

    def test_remove_compounds_list(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compounds(compounds=["x", "y"])
        self.assertEqual(model.compounds, ["z"])

    def test_remove_compounds_set(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compounds(compounds={"x", "y"})
        self.assertEqual(model.compounds, ["z"])

    def test_remove_compounds_dict(self):
        model = Model()
        model.add_compounds(compounds=["x", "y", "z"])
        model.remove_compounds(compounds={"x": 1, "y": 1})
        self.assertEqual(model.compounds, ["z"])

    ############################################################################
    # Updating meta info
    ############################################################################

    def test_update_compound_meta_info(self):
        model = Model()
        model.add_compound(compound="x")
        model.update_compound_meta_info(compound="x", meta_info={"common_name": "X"})
        self.assertEqual(model.compounds, ["x"])
        self.assertEqual(model.meta_info["compounds"]["x"].common_name, "X")

    def test_update_compound_meta_info_replacing(self):
        model = Model()
        model.add_compound(compound="x", **{"common_name": "X1"})
        model.update_compound_meta_info(compound="x", meta_info={"common_name": "X2"})
        self.assertEqual(model.meta_info["compounds"]["x"].common_name, "X2")

    def test_update_compound_meta_info_additional(self):
        model = Model()
        model.add_compound(compound="x", **{"common_name": "X"})
        model.update_compound_meta_info(compound="x", meta_info={"compartment": "e"})
        self.assertEqual(model.meta_info["compounds"]["x"].common_name, "X")
        self.assertEqual(model.meta_info["compounds"]["x"].compartment, "e")

    ############################################################################
    # Getting compounds
    ############################################################################

    def test_get_compounds(self):
        model = Model()
        model.add_compounds(compounds=("x", "y", "z"))
        self.assertEqual(model.compounds, model.get_compounds())


class SBMLTests(unittest.TestCase):
    def test_create_sbml_compounds_without_meta_info(self):
        model = Model()
        model.add_compound("x")
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_compounds(sbml_model=sbml_model)

        cpd = sbml_model.getListOfSpecies()[0]
        self.assertEqual(cpd.getId(), "x")
        self.assertEqual(cpd.getName(), "")
        self.assertEqual(cpd.getCompartment(), "c")
        self.assertEqual(cpd.getCharge(), 0)
        self.assertEqual(cpd.getPlugin("fbc").getChemicalFormula(), "")
        self.assertEqual(cpd.getConstant(), False)
        self.assertEqual(cpd.getBoundaryCondition(), False)

    def test_create_sbml_compounds_with_meta_info(self):
        model = Model()
        model.add_compound(
            "x",
            **{
                "common_name": "Glucose",
                "charge": -2.0,
                "compartment": "e",
                "formula": "C6H12O6",
            },
        )

        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_compounds(sbml_model=sbml_model)

        cpd = sbml_model.getListOfSpecies()[0]
        self.assertEqual(cpd.getId(), "x")
        self.assertEqual(cpd.getName(), "Glucose")
        self.assertEqual(cpd.getCompartment(), "e")
        self.assertEqual(cpd.getPlugin("fbc").getCharge(), -2)
        self.assertEqual(cpd.getPlugin("fbc").getChemicalFormula(), "C6H12O6")
        self.assertEqual(cpd.getConstant(), False)
        self.assertEqual(cpd.getBoundaryCondition(), False)
