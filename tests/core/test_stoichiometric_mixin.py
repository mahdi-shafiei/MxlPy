# type: ignore

from __future__ import annotations

import unittest

import numpy as np

from modelbase2.ode import Model


class ModelBasicTests(unittest.TestCase):
    def test_init_empty(self):
        model = Model()
        self.assertEqual(model.compounds, [])
        self.assertEqual(model.stoichiometries, {})
        self.assertEqual(model.stoichiometries_by_compounds, {})

    def test_enter(self):
        pass

    def test_exit(self):
        pass

    def test_copy(self):
        pass


class ModelWarningsTests(unittest.TestCase):
    def test_warn_on_stoichiometry_replacement(self):
        model = Model()
        model.add_compounds(compounds=["x", "y"])
        model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})
        with self.assertWarns(UserWarning):
            model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})

    def test_warn_on_stoichiometry_by_compounds_replacement(self):
        model = Model()
        model.add_compounds(compounds=["x"])
        model.add_stoichiometry_by_compound(compound="x", stoichiometry={"v1": -1})
        with self.assertWarns(UserWarning):
            model.add_stoichiometry_by_compound(compound="x", stoichiometry={"v1": -1})


class ModelTests(unittest.TestCase):
    """Tests for stoichiometry methods"""

    def test_init(self):
        model = Model(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        self.assertEqual(model.stoichiometries["v1"], {"x": -1, "y": 1})
        self.assertEqual(model.stoichiometries["v2"], {"x": 1, "y": -1})
        self.assertEqual(model.stoichiometries_by_compounds["x"], {"v1": -1, "v2": 1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v1": 1, "v2": -1})

    def test_add_stoichiometry(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(model.stoichiometries_by_compounds["x"], {"v1": -1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v1": 1})

    def test_add_stoichiometry_by_compound(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometry_by_compound(compound="x", stoichiometry={"v1": -1})
        model.add_stoichiometry_by_compound(compound="y", stoichiometry={"v1": 1})
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(model.stoichiometries_by_compounds["x"], {"v1": -1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v1": 1})

    def test_add_stoichiometries(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        self.assertEqual(model.stoichiometries["v1"], {"x": -1, "y": 1})
        self.assertEqual(model.stoichiometries["v2"], {"x": 1, "y": -1})
        self.assertEqual(model.stoichiometries_by_compounds["x"], {"v1": -1, "v2": 1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v1": 1, "v2": -1})

    def test_add_stoichiometries_by_compounds(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries_by_compounds(
            compound_stoichiometries={
                "x": {"v1": -1, "v2": 1},
                "y": {"v1": 1, "v2": -1},
            }
        )
        self.assertEqual(model.stoichiometries["v1"], {"x": -1, "y": 1})
        self.assertEqual(model.stoichiometries["v2"], {"x": 1, "y": -1})
        self.assertEqual(model.stoichiometries_by_compounds["x"], {"v1": -1, "v2": 1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v1": 1, "v2": -1})

    def test_remove_rate_stoichiometry(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        model.remove_rate_stoichiometry(rate_name="v1")
        self.assertEqual(model.stoichiometries["v2"], {"x": 1, "y": -1})
        self.assertEqual(model.stoichiometries_by_compounds["x"], {"v2": 1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v2": -1})

    def test_remove_rate_stoichiometries(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        model.remove_rate_stoichiometries(rate_names=("v1", "v2"))
        self.assertEqual(model.stoichiometries, {})
        self.assertEqual(model.stoichiometries_by_compounds, {})

    def test_remove_compound_stoichiometry(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        model.remove_compound_stoichiometry(compound="x")
        self.assertEqual(model.stoichiometries["v1"], {"y": 1})
        self.assertEqual(model.stoichiometries["v2"], {"y": -1})
        self.assertEqual(model.stoichiometries_by_compounds["y"], {"v1": 1, "v2": -1})

    def test_remove_compound_stoichiometries(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        model.remove_compound_stoichiometries(compounds=("x", "y"))
        self.assertEqual(model.stoichiometries, {})
        self.assertEqual(model.stoichiometries_by_compounds, {})

    def test_get_rate_stoichiometry(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})
        self.assertEqual(
            model.get_rate_stoichiometry(rate_name="v1"), {"x": -1, "y": 1}
        )

    def test_get_compound_stoichiometry(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})
        self.assertEqual(model.get_compound_stoichiometry(compound="x"), {"v1": -1})
        self.assertEqual(model.get_compound_stoichiometry(compound="y"), {"v1": 1})

    def test_get_stoichiometries(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        self.assertEqual(model.stoichiometries, model.get_stoichiometries())

    def test_get_stoichiometries_by_compounds(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometries(
            rate_stoichiometries={"v1": {"x": -1, "y": 1}, "v2": {"x": 1, "y": -1}}
        )
        self.assertEqual(
            model.stoichiometries_by_compounds, model.get_stoichiometries_by_compounds()
        )

    def test_get_stoichiometric_matrix(self):
        model = Model()
        model.add_compounds(("x", "y", "z"))
        model.add_stoichiometries_by_compounds(
            compound_stoichiometries={
                "x": {"v1": 1, "v2": -1},
                "y": {"v2": 1, "v3": -1},
                "z": {"v3": 1, "v4": -1},
            }
        )

        expected = np.array(
            [[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
        )
        np.testing.assert_array_equal(model.get_stoichiometric_matrix(), expected)

    def test_get_stoichiometric_df(self):
        model = Model()
        model.add_compounds(("x", "y", "z"))
        model.add_stoichiometries_by_compounds(
            compound_stoichiometries={
                "x": {"v1": 1, "v2": -1},
                "y": {"v2": 1, "v3": -1},
                "z": {"v3": 1, "v4": -1},
            }
        )

        expected = np.array(
            [[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
        )
        df = model.get_stoichiometric_df()
        np.testing.assert_array_equal(df.values, expected)
        self.assertEqual(df.index.to_list(), ["x", "y", "z"])
        self.assertEqual(df.columns.to_list(), ["v1", "v2", "v3", "v4"])

    def test_get_stoichiometric_matrix_sorted(self):
        model = Model()
        model.add_compounds(("x", "y", "z"))
        model.add_stoichiometries_by_compounds(
            compound_stoichiometries={
                "x": {"v1": 1, "v2": -1},
                "y": {"v2": 1, "v3": -1},
                "z": {"v3": 1, "v4": -1},
            }
        )
        model.remove_rate_stoichiometry(rate_name="v2")
        model.add_stoichiometry(rate_name="v2", stoichiometry={"x": -1, "y": 1})

        expected = np.array(
            [[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
        )
        np.testing.assert_array_equal(model.get_stoichiometric_matrix(), expected)

    def test_get_stoichiometric_df_sorted(self):
        model = Model()
        model.add_compounds(("x", "y", "z"))
        model.add_stoichiometries_by_compounds(
            compound_stoichiometries={
                "x": {"v1": 1, "v2": -1},
                "y": {"v2": 1, "v3": -1},
                "z": {"v3": 1, "v4": -1},
            }
        )
        model.remove_rate_stoichiometry(rate_name="v2")
        model.add_stoichiometry(rate_name="v2", stoichiometry={"x": -1, "y": 1})

        expected = np.array(
            [[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
        )
        df = model.get_stoichiometric_df()
        np.testing.assert_array_equal(df.values, expected)
        self.assertEqual(df.index.to_list(), ["x", "y", "z"])
        self.assertEqual(df.columns.to_list(), ["v1", "v2", "v3", "v4"])


class SourceCodeTests(unittest.TestCase):
    def test_generate_stoichiometries_source_code(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})
        self.assertEqual(
            model._generate_stoichiometries_source_code(),
            "m.add_stoichiometries(rate_stoichiometries={'v1': {'x': -1, 'y': 1}})",
        )


class SBMLTests(unittest.TestCase):
    def test_create_sbml_stoichiometries(self):
        model = Model()
        model.add_compounds(compounds=("x", "y"))
        model.add_stoichiometry(rate_name="v1", stoichiometry={"x": -1, "y": 1})
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_stoichiometries(sbml_model=sbml_model)
        rxn = sbml_model.getReaction("v1")
        self.assertEqual(rxn.getId(), "v1")
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "x")
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "y")
