# type: ignore

from __future__ import annotations

import unittest

import numpy as np

from modelbase2.ode import LinearLabelModel


class CreationTests(unittest.TestCase):
    def test_add_compound(self):
        label_model = LinearLabelModel()
        label_model.add_compound(compound="x", num_labels=3)
        self.assertEqual(label_model.compounds, ["x__0", "x__1", "x__2"])
        self.assertEqual(label_model.isotopomers, {"x": ["x__0", "x__1", "x__2"]})

    def test_add_compound_replacing(self):
        label_model = LinearLabelModel()
        label_model.add_compound(compound="x", num_labels=3)
        with self.assertWarns(UserWarning):
            label_model.add_compound(compound="x", num_labels=1)
        self.assertEqual(label_model.compounds, ["x__0"])
        self.assertEqual(label_model.isotopomers, {"x": ["x__0"]})

    def test_add_compound_fail_on_no_labels(self):
        label_model = LinearLabelModel()
        with self.assertRaises(ValueError):
            label_model.add_compound(compound="x", num_labels=0)

    def test_add_compounds(self):
        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 1, "y": 2})
        self.assertEqual(label_model.compounds, ["x__0", "y__0", "y__1"])
        self.assertEqual(
            label_model.isotopomers, {"x": ["x__0"], "y": ["y__0", "y__1"]}
        )

    def test_add_rate(self):
        label_model = LinearLabelModel()
        label_model.add_rate(rate_name="v1__1", base_name="v1", substrate="x")
        rate = label_model.rates["v1__1"]
        self.assertEqual(rate["base_name"], "v1")
        self.assertEqual(rate["substrate"], "x")

    def test_add_rate_replacing(self):
        label_model = LinearLabelModel()
        label_model.add_rate(rate_name="v1__1", base_name="v1", substrate="x")
        with self.assertWarns(UserWarning):
            label_model.add_rate(rate_name="v1__1", base_name="v1", substrate="y")
            rate = label_model.rates["v1__1"]
            self.assertEqual(rate["base_name"], "v1")
            self.assertEqual(rate["substrate"], "y")

    def test_add_reaction(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )

        self.assertEqual(
            label_model.stoichiometries,
            {"v1__0": {"x__0": -1, "y__0": 1}, "v1__1": {"x__1": -1, "y__1": 1}},
        )
        rate = label_model.rates["v1__0"]
        self.assertEqual(rate["base_name"], "v1")
        self.assertEqual(rate["substrate"], "x__0")

        rate = label_model.rates["v1__1"]
        self.assertEqual(rate["base_name"], "v1")
        self.assertEqual(rate["substrate"], "x__1")

    def test_add_reaction_replacing(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2, "y": 2})
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2, "y": 2, "z": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )
        with self.assertWarns(UserWarning):
            label_model.add_reaction(
                rate_name="v1", stoichiometry={"x": -1, "z": 1}, labelmap=[0, 1]
            )

        self.assertEqual(
            label_model.stoichiometries,
            {"v1__0": {"x__0": -1, "z__0": 1}, "v1__1": {"x__1": -1, "z__1": 1}},
        )
        rate = label_model.rates["v1__0"]
        self.assertEqual(rate["base_name"], "v1")
        self.assertEqual(rate["substrate"], "x__0")

        rate = label_model.rates["v1__1"]
        self.assertEqual(rate["base_name"], "v1")
        self.assertEqual(rate["substrate"], "x__1")

    def test_add_reaction_influx(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 1})
        with self.assertWarns(UserWarning):
            label_model.add_reaction(
                rate_name="v1", stoichiometry={"x": 1}, labelmap=[0]
            )
        self.assertEqual(label_model.stoichiometries, {"v1__0": {"EXT": -1, "x__0": 1}})

    def test_add_reaction_outflux(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 1})
        label_model.add_reaction(rate_name="v1", stoichiometry={"x": -1}, labelmap=[0])
        self.assertEqual(label_model.stoichiometries, {"v1__0": {"x__0": -1, "EXT": 1}})

    def test_add_reaction_fail_on_missing_labels(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2})
        with self.assertRaises(ValueError):
            label_model.add_reaction(
                rate_name="v1", stoichiometry={"x": -1}, labelmap=[0]
            )


class GenerateY0Test(unittest.TestCase):
    def test_generate_y0(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2})
        self.assertEqual(label_model.generate_y0(), {"x__0": 0, "x__1": 0})

    def test_generate_y0_single(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2})
        self.assertEqual(
            label_model.generate_y0(initial_labels={"x": 0}), {"x__0": 1.0, "x__1": 0}
        )

    def test_generate_y0_distributed(self):
        label_model = LinearLabelModel()
        label_model.add_compounds({"x": 2})
        self.assertEqual(
            label_model.generate_y0(initial_labels={"x": [0, 1]}),
            {"x__0": 0.5, "x__1": 0.5},
        )


class SimulationTests(unittest.TestCase):
    def test_get_fluxes(self):
        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )
        y0 = label_model.generate_y0({"x": 0})
        v_ss = {"v1": 1}
        self.assertEqual(
            label_model._get_fluxes(fcd=y0, v_ss=v_ss), {"v1__0": 1.0, "v1__1": 0}
        )

    def test_get_fluxes_reverse(self):
        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[1, 0]
        )
        y0 = label_model.generate_y0({"x": 0})
        v_ss = {"v1": 1}

        self.assertEqual(
            label_model._get_fluxes(fcd=y0, v_ss=v_ss), {"v1__0": 0, "v1__1": 1.0}
        )

    def test_get_fluxes_dict_array_df(self):
        v_ss = {"v1": 1}
        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )
        y0 = label_model.generate_y0({"x": 0})

        # non-dict input
        self.assertEqual(
            label_model.get_fluxes_dict(y=list(y0.values()), v_ss=v_ss),
            {"v1__0": 1.0, "v1__1": 0},
        )

        # dict
        self.assertEqual(
            label_model.get_fluxes_dict(y=y0, v_ss=v_ss), {"v1__0": 1.0, "v1__1": 0}
        )

        # array
        np.testing.assert_array_equal(
            label_model.get_fluxes_array(y=y0, v_ss=v_ss), [1, 0]
        )

        # df
        df = label_model.get_fluxes_df(y=y0, v_ss=v_ss)
        np.testing.assert_array_equal(df.index.values, [0])
        np.testing.assert_array_equal(df.columns.values, ["v1__0", "v1__1"])
        np.testing.assert_array_equal(df.values, [[1.0, 0.0]])

    def test_get_rhs(self):
        v_ss = {"v1": 1}
        y_ss = {"x": 2, "y": 4}

        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )
        y0 = list(label_model.generate_y0({"x": 0}).values())
        label_model._v_ss = v_ss
        label_model._y_ss = y_ss
        label_model._external_label = 1
        self.assertEqual(label_model._get_rhs(0, y0), [-0.5, 0.0, 0.25, 0.0])

    def test_get_rhs_reverse(self):
        v_ss = {"v1": 1}
        y_ss = {"x": 2, "y": 4}

        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[1, 0]
        )
        y0 = list(label_model.generate_y0({"x": 0}).values())
        label_model._v_ss = v_ss
        label_model._y_ss = y_ss
        label_model._external_label = 1
        self.assertEqual(label_model._get_rhs(0, y0), [-0.5, 0.0, 0, 0.25])

    def test_get_right_hand_side(self):
        v_ss = {"v1": 1}
        y_ss = {"x": 2, "y": 4}

        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )
        y0 = label_model.generate_y0({"x": 0})
        # dict input
        self.assertEqual(
            label_model.get_right_hand_side(y_labels=y0, y_ss=y_ss, v_ss=v_ss),
            {"x__0": -0.5, "x__1": 0.0, "y__0": 0.25, "y__1": 0.0},
        )
        # list input
        self.assertEqual(
            label_model.get_right_hand_side(
                y_labels=list(y0.values()), y_ss=y_ss, v_ss=v_ss
            ),
            {"x__0": -0.5, "x__1": 0.0, "y__0": 0.25, "y__1": 0.0},
        )


class SBMLTests(unittest.TestCase):
    def test_create_sbml_reactions(self):
        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )

        doc = label_model._create_sbml_document()
        sbml_model = label_model._create_sbml_model(doc=doc)
        label_model._create_sbml_reactions(sbml_model=sbml_model)
        rxn = sbml_model.getReaction("v1__0")
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "x__0")
        self.assertEqual(rxn.getListOfReactants()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "y__0")
        self.assertEqual(rxn.getListOfProducts()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getKineticLaw().getFormula(), "v1 * x__0")

        rxn = sbml_model.getReaction("v1__1")
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "x__1")
        self.assertEqual(rxn.getListOfReactants()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "y__1")
        self.assertEqual(rxn.getListOfProducts()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getKineticLaw().getFormula(), "v1 * x__1")

    def test_model_to_sbml(self):
        label_model = LinearLabelModel()
        label_model.add_compounds(compounds={"x": 2, "y": 2})
        label_model.add_reaction(
            rate_name="v1", stoichiometry={"x": -1, "y": 1}, labelmap=[0, 1]
        )
        label_model._model_to_sbml()
        self.assertTrue(True)
