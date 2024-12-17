# type: ignore

from __future__ import annotations

import unittest

from modelbase.ode import Model
from modelbase.ode import ratelaws as rl


class RateLawTests(unittest.TestCase):
    def test_constant(self):
        ratelaw = rl.Constant(product="x1", k="k_in")
        self.assertEqual(ratelaw.get_sbml_function_string(), "k_in")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "constant")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "constant")
        self.assertEqual(rate["parameters"], ["k_in"])
        self.assertEqual(rate["substrates"], [])
        self.assertEqual(rate["products"], ["x1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], [])
        self.assertEqual(rate["reversible"], False)

    def test_mass_action_1_1(self):
        ratelaw = rl.MassAction(substrates=["x1"], products=["y1"], k_fwd="k1")
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "mass_action_1")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_1")
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1"])
        self.assertEqual(rate["reversible"], False)

    def test_mass_action_2_1(self):
        ratelaw = rl.MassAction(substrates=["x1", "x2"], products=["y1"], k_fwd="k1")
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1 * x2")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "mass_action_2")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_2")
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["x1", "x2"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "x2"])
        self.assertEqual(rate["reversible"], False)

    def test_mass_action_1_2(self):
        ratelaw = rl.MassAction(substrates=["x1"], products=["y1", "y2"], k_fwd="k1")
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "mass_action_1")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_1")
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1", "y2"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1"])
        self.assertEqual(rate["reversible"], False)

    def test_mass_action_2_2(self):
        ratelaw = rl.MassAction(
            substrates=["x1", "x2"], products=["y1", "y2"], k_fwd="k1"
        )
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1 * x2")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "mass_action_2")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_2")
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["x1", "x2"])
        self.assertEqual(rate["products"], ["y1", "y2"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "x2"])
        self.assertEqual(rate["reversible"], False)

    def test_mass_action_variable(self):
        ratelaw = rl.MassAction(
            substrates=["x1", "x2", "x3", "x4", "x5"],
            products=["y1", "y2", "y3", "y4", "y5"],
            k_fwd="k1",
        )
        self.assertEqual(
            ratelaw.get_sbml_function_string(),
            "k1 * x1 * x2 * x3 * x4 * x5",
        )
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "mass_action_variable")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_variable")
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["x1", "x2", "x3", "x4", "x5"])
        self.assertEqual(rate["products"], ["y1", "y2", "y3", "y4", "y5"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "x2", "x3", "x4", "x5"])
        self.assertEqual(rate["reversible"], False)

    def test_reversible_mass_action_1_1(self):
        ratelaw = rl.ReversibleMassAction(
            substrates=["x1"], products=["y1"], k_fwd="k1", k_bwd="k2"
        )
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1 - k2 * y1")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_mass_action_1_1")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_mass_action_1_1")
        self.assertEqual(rate["parameters"], ["k1", "k2"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "y1"])
        self.assertEqual(rate["reversible"], True)

    def test_reversible_mass_action_2_1(self):
        ratelaw = rl.ReversibleMassAction(
            substrates=["x1", "x2"], products=["y1"], k_fwd="k1", k_bwd="k2"
        )
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1 * x2 - k2 * y1")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_mass_action_2_1")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_mass_action_2_1")
        self.assertEqual(rate["parameters"], ["k1", "k2"])
        self.assertEqual(rate["substrates"], ["x1", "x2"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "x2", "y1"])
        self.assertEqual(rate["reversible"], True)

    def test_reversible_mass_action_1_2(self):
        ratelaw = rl.ReversibleMassAction(
            substrates=["x1"], products=["y1", "y2"], k_fwd="k1", k_bwd="k2"
        )
        self.assertEqual(ratelaw.get_sbml_function_string(), "k1 * x1 - k2 * y1 * y2")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_mass_action_1_2")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_mass_action_1_2")
        self.assertEqual(rate["parameters"], ["k1", "k2"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1", "y2"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "y1", "y2"])
        self.assertEqual(rate["reversible"], True)

    def test_reversible_mass_action_2_2(self):
        ratelaw = rl.ReversibleMassAction(
            substrates=["x1", "x2"], products=["y1", "y2"], k_fwd="k1", k_bwd="k2"
        )
        self.assertEqual(
            ratelaw.get_sbml_function_string(), "k1 * x1 * x2 - k2 * y1 * y2"
        )
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_mass_action_2_2")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_mass_action_2_2")
        self.assertEqual(rate["parameters"], ["k1", "k2"])
        self.assertEqual(rate["substrates"], ["x1", "x2"])
        self.assertEqual(rate["products"], ["y1", "y2"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "x2", "y1", "y2"])
        self.assertEqual(rate["reversible"], True)

    def test_reversible_mass_action_variable(self):
        ratelaw = rl.ReversibleMassAction(
            substrates=["x1", "x2", "x3", "x4", "x5"],
            products=["y1", "y2", "y3", "y4", "y5"],
            k_fwd="k1",
            k_bwd="k2",
        )
        self.assertEqual(
            ratelaw.get_sbml_function_string(),
            "k1 * x1 * x2 * x3 * x4 * x5 - k2 * y1 * y2 * y3 * y4 * y5",
        )
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_mass_action_variable_5")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_mass_action_variable_5")
        self.assertEqual(rate["parameters"], ["k1", "k2"])
        self.assertEqual(rate["substrates"], ["x1", "x2", "x3", "x4", "x5"])
        self.assertEqual(rate["products"], ["y1", "y2", "y3", "y4", "y5"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(
            rate["dynamic_variables"],
            ["x1", "x2", "x3", "x4", "x5", "y1", "y2", "y3", "y4", "y5"],
        )
        self.assertEqual(rate["reversible"], True)

    def test_michaelis_menten(self):
        ratelaw = rl.MichaelisMenten(S="x1", P="y1", vmax="vmax1", km="km1")
        self.assertEqual(ratelaw.get_sbml_function_string(), "x1 * vmax1 / (x1 + km1)")
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "michaelis_menten")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "michaelis_menten")
        self.assertEqual(rate["parameters"], ["vmax1", "km1"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1"])
        self.assertEqual(rate["reversible"], False)

    def test_reversible_michaelis_menten(self):
        ratelaw = rl.ReversibleMichaelisMenten(
            S="x1",
            P="y1",
            vmax_fwd="vmax1",
            km_fwd="km1",
            vmax_bwd="vmax2",
            km_bwd="km2",
        )
        self.assertEqual(
            ratelaw.get_sbml_function_string(),
            "(vmax1 * x1 / km1 - vmax2 * y1 / km2) / (1 + x1 / km1 + y1 / km2)",
        )
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_michaelis_menten")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_michaelis_menten")
        self.assertEqual(rate["parameters"], ["vmax1", "vmax2", "km1", "km2"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "y1"])
        self.assertEqual(rate["reversible"], True)

    def test_reversible_michaelis_menten_keq(self):
        ratelaw = rl.ReversibleMichaelisMentenKeq(
            S="x1",
            P="y1",
            vmax_fwd="vmax1",
            km_fwd="km1",
            k_eq="keq1",
            km_bwd="km2",
        )
        self.assertEqual(
            ratelaw.get_sbml_function_string(),
            "vmax1 / km1 * (x1 - y1 / keq1)/ (1 + x1 / km1 + y1 / km2)",
        )
        function = ratelaw.get_rate_function()
        self.assertEqual(function.__name__, "reversible_michaelis_menten_keq")

        model = Model()
        model.add_reaction_from_ratelaw("v1", ratelaw=ratelaw)
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "reversible_michaelis_menten_keq")
        self.assertEqual(rate["parameters"], ["vmax1", "km1", "km2", "keq1"])
        self.assertEqual(rate["substrates"], ["x1"])
        self.assertEqual(rate["products"], ["y1"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x1", "y1"])
        self.assertEqual(rate["reversible"], True)
