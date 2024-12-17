# type: ignore

from __future__ import annotations

import unittest

import numpy as np
import pytest
from modelbase.ode import Model
from modelbase.ode import ratelaws as rl


def test_str(multiline_comparison):
    model = Model()
    model.add_compounds(("x", "y"))
    model.add_stoichiometry("v1", {"x": -1, "y": 1})
    multiline_comparison(("Model:", "    2 Compounds", "    1 Reactions"), str(model))


class ModelBaseTests(unittest.TestCase):
    def test_init_empty(self):
        model = Model()
        self.assertTrue(model)

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


class ReactionTests(unittest.TestCase):
    def test_add_reaction(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, k: k * x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=["k"],
            reversible=False,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x"])
        self.assertFalse(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_add_reaction_modifier(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y", "xi"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, xi, k: k / xi * x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=["xi"],
            parameters=["k"],
            reversible=False,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], ["xi"])
        self.assertEqual(rate["dynamic_variables"], ["x", "xi"])
        self.assertFalse(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_add_reaction_modifier_time(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, time, k: k / time * x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=["time"],
            parameters=["k"],
            reversible=False,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], ["time"])
        self.assertEqual(rate["dynamic_variables"], ["x", "time"])
        self.assertFalse(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_add_reaction_reversible(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, y, k: k * (x - y),
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=["k"],
            reversible=True,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["x", "y"])
        self.assertTrue(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_add_reaction_modifier_reversible(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y", "xi"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, y, xi, k: k / xi * (x - y),
            stoichiometry={"x": -1, "y": 1},
            modifiers=["xi"],
            parameters=["k"],
            reversible=True,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], ["xi"])
        self.assertEqual(rate["dynamic_variables"], ["x", "y", "xi"])
        self.assertTrue(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_add_reaction_modifier_time_reversible(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y", "xi"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, y, time, k: k / time * (x - y),
            stoichiometry={"x": -1, "y": 1},
            modifiers=["time"],
            parameters=["k"],
            reversible=True,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], ["time"])
        self.assertEqual(rate["dynamic_variables"], ["x", "y", "time"])
        self.assertTrue(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_add_reaction_from_ratelaw(self):
        model = Model()
        model.add_reaction_from_ratelaw(
            rate_name="v1",
            ratelaw=rl.MassAction(substrates=["X"], products=["Y"], k_fwd="k2"),
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_1")
        self.assertEqual(rate["parameters"], ["k2"])
        self.assertEqual(rate["substrates"], ["X"])
        self.assertEqual(rate["products"], ["Y"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["X"])
        self.assertEqual(rate["reversible"], False)
        self.assertEqual(model.meta_info["rates"]["v1"].sbml_function, "k2 * X")

    def test_add_reaction_from_ratelaw_meta_info(self):
        model = Model()
        model.add_reaction_from_ratelaw(
            rate_name="v1",
            ratelaw=rl.MassAction(substrates=["X"], products=["Y"], k_fwd="k2"),
            **{"sbml_function": "nonsense"},
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "mass_action_1")
        self.assertEqual(rate["parameters"], ["k2"])
        self.assertEqual(rate["substrates"], ["X"])
        self.assertEqual(rate["products"], ["Y"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["X"])
        self.assertEqual(rate["reversible"], False)
        self.assertEqual(model.meta_info["rates"]["v1"].sbml_function, "nonsense")

    def test_add_reaction_dynamic_variables(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y", "z"])
        model.add_reaction(
            rate_name="v1",
            function=lambda z, k: k * z,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            dynamic_variables=["z"],
            parameters=["k"],
            reversible=False,
        )
        rate = model.rates["v1"]
        self.assertEqual(rate["parameters"], ["k"])
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], [])
        self.assertEqual(rate["dynamic_variables"], ["z"])
        self.assertFalse(rate["reversible"])
        self.assertEqual(model.stoichiometries, {"v1": {"x": -1, "y": 1}})
        self.assertEqual(
            model.stoichiometries_by_compounds, {"x": {"v1": -1}, "y": {"v1": 1}}
        )

    def test_update_reaction(self):
        parameters = {"k1": 1, "k2": 2}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, k: k * x,
            stoichiometry={"x1": -1, "y1": 1},
            modifiers=None,
            parameters=["k1"],
            reversible=False,
        )
        model.update_reaction(
            "v1",
            function=None,
            stoichiometry={"x2": -1, "y2": 1},
            modifiers=["Z"],
            parameters=["k2"],
            reversible=True,
        )

        rate = model.rates["v1"]
        self.assertEqual(rate["function"].__name__, "v1")
        self.assertEqual(rate["parameters"], ["k2"])
        self.assertEqual(rate["substrates"], ["x2"])
        self.assertEqual(rate["products"], ["y2"])
        self.assertEqual(rate["modifiers"], ["Z"])
        self.assertEqual(rate["dynamic_variables"], ["x2", "y2", "Z"])
        self.assertEqual(rate["reversible"], True)
        self.assertEqual(model.stoichiometries["v1"], {"x2": -1, "y2": 1})

    # def test_update_reaction_meta_info(self):
    #     parameters = {"k1": 1, "k2": 2}
    #     model = Model(parameters=parameters)
    #     model.add_compounds(["x", "y"])
    #     model.add_reaction(
    #         rate_name="v1",
    #         function=lambda x, k: k * x,
    #         stoichiometry={"x": -1, "y": 1},
    #         modifiers=None,
    #         parameters=["k1"],
    #         reversible=False,
    #     )
    #     model.update_reaction("v1", **{"sbml_function": "k1 * x"})
    #     rate = model.rates["v1"]
    #     self.assertEqual(rate["function"].__name__, "v1")
    #     self.assertEqual(rate["parameters"], ["k1"])
    #     self.assertEqual(rate["substrates"], ["x"])
    #     self.assertEqual(rate["products"], ["y"])
    #     self.assertEqual(rate["modifiers"], [])
    #     self.assertEqual(rate["dynamic_variables"], ["x"])
    #     self.assertEqual(rate["reversible"], False)
    #     self.assertEqual(model.meta_info["rates"]["v1"].sbml_function, "k1 * x")

    def test_remove_reaction(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, k: k * x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=["k"],
            reversible=False,
        )
        model.remove_reaction(rate_name="v1")
        self.assertEqual(model.stoichiometries, {})
        self.assertEqual(model.stoichiometries_by_compounds, {})

    def test_remove_reactions(self):
        parameters = {"k": 1}
        model = Model(parameters=parameters)
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, k: k * x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=["k"],
            reversible=False,
        )
        model.add_reaction(
            rate_name="v1_rev",
            function=lambda y, k: k * y,
            stoichiometry={"y": -1, "x": 1},
            modifiers=None,
            parameters=["k"],
            reversible=False,
        )
        model.remove_reactions(rate_names=("v1", "v1_rev"))
        self.assertEqual(model.stoichiometries, {})
        self.assertEqual(model.stoichiometries_by_compounds, {})


# Algebraic modules
def rapid_equilibrium(substrate, k_eq):
    x = substrate / (1 + k_eq)
    y = substrate * k_eq / (1 + k_eq)
    return x, y


# Rates
def constant(k):
    return k


def time_dependent_mass_action(S, time, k_deg):
    return np.exp(-k_deg * S * time)


def mass_action(S, k):
    return k * S


def reversible_mass_action(S, P, kf, kr):
    return kf * S - kr * P


def create_toy_model():
    parameters = {"k_in": 2, "k_eq": 3, "kf": 1, "kr": 1, "k_deg": 1, "k_out": 1}
    model = Model(parameters=parameters)
    model.add_compounds(["A", "x1", "y1"])
    model.add_algebraic_module(
        module_name="RE",
        function=rapid_equilibrium,
        compounds=["A"],
        derived_compounds=["x0", "y0"],
        modifiers=None,
        parameters=["k_eq"],
    )
    model.add_reaction(
        rate_name="influx",
        function=constant,
        stoichiometry={"A": 1},
        modifiers=None,
        parameters=["k_in"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="v1",
        function=reversible_mass_action,
        stoichiometry={"x0": -1, "x1": 1},
        modifiers=None,
        parameters=["kf", "kr"],
        reversible=True,
    )
    model.add_reaction(
        rate_name="v2",
        function=time_dependent_mass_action,
        stoichiometry={"y0": -1, "y1": 1},
        modifiers=["time"],
        parameters=["k_deg"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="outflux0",
        function=mass_action,
        stoichiometry={"A": -1},
        modifiers=None,
        parameters=["k_out"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="outflux1",
        function=mass_action,
        stoichiometry={"x1": -1},
        modifiers=None,
        parameters=["k_out"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="outflux2",
        function=mass_action,
        stoichiometry={"y1": -1},
        modifiers=None,
        parameters=["k_out"],
        reversible=False,
    )
    return model


class SimulationFunctionTests(unittest.TestCase):
    def test_get_full_concentration_dict_with_list(self):
        model = create_toy_model()
        y = [1, 2, 3]
        fcd = model.get_full_concentration_dict(y=y, t=0)
        self.assertEqual(fcd["A"], 1)
        self.assertEqual(fcd["x1"], 2)
        self.assertEqual(fcd["y1"], 3)
        self.assertEqual(fcd["time"], 0)
        self.assertEqual(fcd["x0"], 0.25)
        self.assertEqual(fcd["y0"], 0.75)

    def test_get_full_concentration_dict_with_dict(self):
        model = create_toy_model()
        y = {"A": 1, "x1": 2, "y1": 3}
        y = model.get_full_concentration_dict(y=y, t=0)
        fcd = model.get_full_concentration_dict(y=y, t=0)
        self.assertEqual(fcd["A"], 1)
        self.assertEqual(fcd["x1"], 2)
        self.assertEqual(fcd["y1"], 3)
        self.assertEqual(fcd["time"], 0)
        self.assertEqual(fcd["x0"], 0.25)
        self.assertEqual(fcd["y0"], 0.75)

    def test_get_full_concentration_dict_with_fcd(self):
        model = create_toy_model()
        y = {"A": 1, "x1": 2, "y1": 3}
        fcd = model.get_full_concentration_dict(y=y, t=0)
        self.assertEqual(fcd["A"], 1)
        self.assertEqual(fcd["x1"], 2)
        self.assertEqual(fcd["y1"], 3)
        self.assertEqual(fcd["time"], 0)
        self.assertEqual(fcd["x0"], 0.25)
        self.assertEqual(fcd["y0"], 0.75)

    def test_get_full_concentration_dict_with_simulation_results(self):
        model = create_toy_model()
        t = np.ones((10))
        y = np.ones((10, 3)) * [1, 2, 3]
        fcd = model.get_full_concentration_dict(y=y, t=t)
        self.assertTrue((fcd["A"] == 1).all())
        self.assertTrue((fcd["x1"] == 2).all())
        self.assertTrue((fcd["y1"] == 3).all())
        self.assertTrue((fcd["x0"] == 0.25).all())
        self.assertTrue((fcd["y0"] == 0.75).all())
        np.testing.assert_array_equal(fcd["time"], t)

        self.assertEqual(fcd["A"].shape, (10,))
        self.assertEqual(fcd["x1"].shape, (10,))
        self.assertEqual(fcd["y1"].shape, (10,))
        self.assertEqual(fcd["time"].shape, (10,))
        self.assertEqual(fcd["x0"].shape, (10,))
        self.assertEqual(fcd["y0"].shape, (10,))

    def test_module_shapes(self):
        def singleton_module(x):
            return x * 1

        def singleton_module_no_input():
            return 1

        def tuple_module_1(x):
            return (x * 1,)

        def tuple_module_1_no_input():
            return (1,)

        def tuple_module_2(x):
            return x * 1, x * 2

        def tuple_module_2_no_input():
            return 1, 2

        def tuple_module_3(x):
            return x * 1, x * 2, x * 3

        def tuple_module_3_no_input():
            return 1, 2, 3

        def list_module_1(x):
            return [x * 1]

        def list_module_1_no_input():
            return [1]

        def list_module_2(x):
            return [x * 1, x * 2]

        def list_module_2_no_input():
            return [1, 2]

        def list_module_3(x):
            return [x * 1, x * 2, x * 3]

        def list_module_3_no_input():
            return [1, 2, 3]

        def array_module_1(x):
            return np.array([x * 1])

        def array_module_1_no_input():
            return np.array([1])

        def array_module_2(x):
            return np.array([x * 1, x * 2])

        def array_module_2_no_input():
            return np.array([1, 2])

        def array_module_3(x):
            return np.array([x * 1, x * 2, x * 3])

        def array_module_3_no_input():
            return np.array([1, 2, 3])

        model = Model()
        model.add_compound("x")
        model.add_algebraic_module(
            module_name="singleton",
            function=singleton_module,
            compounds=["x"],
            derived_compounds=["iS1"],
        )
        model.add_algebraic_module(
            module_name="singleton_no_input",
            function=singleton_module_no_input,
            compounds=None,
            derived_compounds=["nS1"],
        )
        model.add_algebraic_module(
            module_name="tuple_1",
            function=tuple_module_1,
            compounds=["x"],
            derived_compounds=[
                "iT1.1",
            ],
        )
        model.add_algebraic_module(
            module_name="tuple_1_no_input",
            function=tuple_module_1_no_input,
            compounds=None,
            derived_compounds=["nT1.1"],
        )
        model.add_algebraic_module(
            module_name="tuple_2",
            function=tuple_module_2,
            compounds=["x"],
            derived_compounds=["iT2.1", "iT2.2"],
        )
        model.add_algebraic_module(
            module_name="tuple_2_no_input",
            function=tuple_module_2_no_input,
            compounds=None,
            derived_compounds=["nT2.1", "nT2.2"],
        )
        model.add_algebraic_module(
            module_name="tuple_3",
            function=tuple_module_3,
            compounds=["x"],
            derived_compounds=["iT3.1", "iT3.2", "iT3.3"],
        )
        model.add_algebraic_module(
            module_name="tuple_3_no_input",
            function=tuple_module_3_no_input,
            compounds=None,
            derived_compounds=["nT3.1", "nT3.2", "nT3.3"],
        )
        model.add_algebraic_module(
            module_name="list_1",
            function=list_module_1,
            compounds=["x"],
            derived_compounds=[
                "iL1.1",
            ],
        )
        model.add_algebraic_module(
            module_name="list_1_no_input",
            function=list_module_1_no_input,
            compounds=None,
            derived_compounds=[
                "nL1.1",
            ],
        )
        model.add_algebraic_module(
            module_name="list_2",
            function=list_module_2,
            compounds=["x"],
            derived_compounds=["iL2.1", "iL2.2"],
        )
        model.add_algebraic_module(
            module_name="list_2_no_input",
            function=list_module_2_no_input,
            compounds=None,
            derived_compounds=["nL2.1", "nL2.2"],
        )
        model.add_algebraic_module(
            module_name="list_3",
            function=list_module_3,
            compounds=["x"],
            derived_compounds=["iL3.1", "iL3.2", "iL3.3"],
        )
        model.add_algebraic_module(
            module_name="list_3_no_input",
            function=list_module_3_no_input,
            compounds=None,
            derived_compounds=["nL3.1", "nL3.2", "nL3.3"],
        )
        model.add_algebraic_module(
            module_name="array_1",
            function=array_module_1,
            compounds=["x"],
            derived_compounds=["iA1.1"],
        )
        model.add_algebraic_module(
            module_name="array_1_no_input",
            function=array_module_1_no_input,
            compounds=None,
            derived_compounds=["nA1.1"],
        )
        model.add_algebraic_module(
            module_name="array_2",
            function=array_module_2,
            compounds=["x"],
            derived_compounds=["iA2.1", "iA2.2"],
        )
        model.add_algebraic_module(
            module_name="array_2_no_input",
            function=array_module_2_no_input,
            compounds=None,
            derived_compounds=["nA2.1", "nA2.2"],
        )
        model.add_algebraic_module(
            module_name="array_3",
            function=array_module_3,
            compounds=["x"],
            derived_compounds=["iA3.1", "iA3.2", "iA3.3"],
        )
        model.add_algebraic_module(
            module_name="array_3_no_input",
            function=array_module_3_no_input,
            compounds=None,
            derived_compounds=["nA3.1", "nA3.2", "nA3.3"],
        )

        fcd = model.get_full_concentration_dict({"x": 1})
        for cpd in model.get_all_compounds():
            self.assertEqual(fcd[cpd].shape, (1,))

        fcd = model.get_full_concentration_dict({"x": [1, 2]}, t=[0, 1])
        for cpd in model.get_all_compounds():
            self.assertEqual(fcd[cpd].shape, (2,))

        fcd = model.get_full_concentration_dict({"x": [1, 2, 3]}, t=[0, 1, 2])
        for cpd in model.get_all_compounds():
            self.assertEqual(fcd[cpd].shape, (3,))

    def test_get_fluxes_with_list(self):
        model = create_toy_model()
        t = 0
        y = [1, 2, 3]
        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["influx"], 2)
        self.assertEqual(fluxes["v1"], -1.75)
        self.assertEqual(fluxes["v2"], 1)
        self.assertEqual(fluxes["outflux0"], 1)
        self.assertEqual(fluxes["outflux1"], 2)
        self.assertEqual(fluxes["outflux2"], 3)

    def test_get_fluxes_with_dict(self):
        model = create_toy_model()
        t = 0
        y = {"A": 1, "x1": 2, "y1": 3}
        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["influx"], 2)
        self.assertEqual(fluxes["v1"], -1.75)
        self.assertEqual(fluxes["v2"], 1)
        self.assertEqual(fluxes["outflux0"], 1)
        self.assertEqual(fluxes["outflux1"], 2)
        self.assertEqual(fluxes["outflux2"], 3)

    def test_get_fluxes_with_fcd(self):
        model = create_toy_model()
        t = 0
        y = {"A": 1, "x1": 2, "y1": 3}
        y = model.get_full_concentration_dict(y=y, t=t)
        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["influx"], 2)
        self.assertEqual(fluxes["v1"], -1.75)
        self.assertEqual(fluxes["v2"], 1)
        self.assertEqual(fluxes["outflux0"], 1)
        self.assertEqual(fluxes["outflux1"], 2)
        self.assertEqual(fluxes["outflux2"], 3)

    def test_get_fluxes_with_simulation_results(self):
        model = create_toy_model()
        # Only the shape is important, the time variable
        # is tested in test_get_fluxes_time
        t = np.zeros(10)
        y = np.ones((10, 3)) * [1, 2, 3]
        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertTrue((fluxes["influx"] == 2).all())
        self.assertTrue((fluxes["v1"] == -1.75).all())
        self.assertTrue((fluxes["v2"] == 1).all())
        self.assertTrue((fluxes["outflux0"] == 1).all())
        self.assertTrue((fluxes["outflux1"] == 2).all())
        self.assertTrue((fluxes["outflux2"] == 3).all())

    def test_get_fluxes_time_single(self):
        model = Model()
        model.add_compound("x")
        model.add_reaction(
            rate_name="v1",
            function=lambda time: time,
            stoichiometry={"x": 1},
            modifiers=["time"],
        )
        fluxes = model.get_fluxes_dict(y=[0], t=0)
        self.assertEqual(fluxes["v1"], 0)

    def test_get_fluxes_time_array(self):
        model = Model()
        model.add_compound("x")
        model.add_reaction(
            rate_name="v1",
            function=lambda time: time,
            stoichiometry={"x": 1},
            modifiers=["time"],
        )
        fluxes = model.get_fluxes_dict(y=[0], t=np.arange(10))
        np.testing.assert_array_equal(fluxes["v1"], np.arange(10))

    def test_get_fluxes_array_single(self):
        model = create_toy_model()
        # Only the shape is important, the time variable
        # is tested in test_get_fluxes_time
        t = np.zeros(10)
        y = np.ones((10, 3)) * [1, 2, 3]
        fluxes = model.get_fluxes_array(y=y, t=t)
        self.assertEqual(fluxes.shape, (10, 6))
        np.testing.assert_array_equal(
            fluxes, np.ones((10, 1)) * np.array([2.0, -1.75, 1.0, 1.0, 2.0, 3.0])
        )

    def test_get_fluxes_array_multiple(self):
        model = create_toy_model()
        t = 0
        y = [1, 2, 3]
        fluxes = model.get_fluxes_array(y=y, t=t)
        self.assertEqual(fluxes.shape, (1, 6))
        np.testing.assert_array_equal(
            fluxes, np.array([[2.0, -1.75, 1.0, 1.0, 2.0, 3.0]])
        )

    def test_get_rhs_stoich_one(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        t = 0
        y = [1]
        rhs = model._get_rhs(t=t, y=y)
        self.assertEqual(rhs[0], -1)
        self.assertEqual(rhs[1], 1)

    def test_get_rhs_stoich_two(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -2, "y": 2},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        t = 0
        y = [1]
        rhs = model._get_rhs(t=t, y=y)
        self.assertEqual(rhs[0], -2)
        self.assertEqual(rhs[1], 2)

    def test_get_rhs_stoich_asymmetric(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -2, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        t = 0
        y = [1]
        rhs = model._get_rhs(t=t, y=y)
        self.assertEqual(rhs[0], -2)
        self.assertEqual(rhs[1], 1)

    def test_get_right_hand_side_list(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        t = 0
        y = [1, 0]
        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -1)
        self.assertEqual(rhs["dydt"], 1)

    def test_get_right_hand_side_dict(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        t = 0
        y = {"x": 1, "y": 0}
        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -1)
        self.assertEqual(rhs["dydt"], 1)

    def test_get_right_hand_side_fcd(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        t = 0
        y = {"x": 1, "y": 0}
        y = model.get_full_concentration_dict(y=y, t=t)
        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -1)
        self.assertEqual(rhs["dydt"], 1)


class StructureChangesTests(unittest.TestCase):
    def test_compound_changes(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )

        # Remove and re-add compound
        model.remove_compound("x")
        model.add_compound("x")

        t = 0
        y = {"x": 2, "y": 0}

        fcd = model.get_full_concentration_dict(y=y, t=t)
        self.assertEqual(fcd["x"], 2)
        self.assertEqual(fcd["y"], 0)
        self.assertEqual(fcd["time"], 0)

        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["v1"], 2)

        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -2)
        self.assertEqual(rhs["dydt"], 2)

    def test_prior_compound_removal(self):
        model = Model()
        model.add_compounds(["A", "x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )

        # Remove and re-add compound
        model.remove_compound("A")

        t = 0
        y = {"x": 2, "y": 0}

        fcd = model.get_full_concentration_dict(y=y, t=t)
        self.assertEqual(fcd["x"], 2)
        self.assertEqual(fcd["y"], 0)
        self.assertEqual(fcd["time"], 0)

        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["v1"], 2)

        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -2)
        self.assertEqual(rhs["dydt"], 2)

    def test_reaction_changes(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )

        # Remove and re-add reaction
        model.remove_reaction("v1")
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )

        t = 0
        y = {"x": 2, "y": 0}

        fcd = model.get_full_concentration_dict(y=y, t=t)
        self.assertEqual(fcd["x"], 2)
        self.assertEqual(fcd["y"], 0)
        self.assertEqual(fcd["time"], 0)

        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["v1"], 2)

        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -2)
        self.assertEqual(rhs["dydt"], 2)

    def test_prior_reaction_removal(self):
        model = Model()
        model.add_compounds(["x", "y"])
        model.add_reaction(
            rate_name="v0",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )
        model.add_reaction(
            rate_name="v1",
            function=lambda x: x,
            stoichiometry={"x": -1, "y": 1},
            modifiers=None,
            parameters=None,
            reversible=False,
        )

        # Remove prior reaction
        model.remove_reaction("v0")

        t = 0
        y = {"x": 2, "y": 0}

        fcd = model.get_full_concentration_dict(y=y, t=t)
        self.assertEqual(fcd["x"], 2)
        self.assertEqual(fcd["y"], 0)
        self.assertEqual(fcd["time"], 0)

        fluxes = model.get_fluxes_dict(y=y, t=t)
        self.assertEqual(fluxes["v1"], 2)

        rhs = model.get_right_hand_side(y=y, t=t)
        self.assertEqual(rhs["dxdt"], -2)
        self.assertEqual(rhs["dydt"], 2)


class ModelConversionTests(unittest.TestCase):
    def test_to_labelmodel(self):
        model = Model()
        model.add_compounds(("X", "Y", "ATP", "ADP", "Z"))
        model.add_reaction(
            rate_name="v1",
            function=lambda x, ATP, z: x * ATP / z,
            stoichiometry={"X": -1, "ATP": -1, "Y": 1, "ADP": 1},
            modifiers=["Z"],
            parameters=["k1"],
        )

        labelcompounds = {"X": 1, "Y": 1, "Z": 1}
        labelmaps = {"v1": [0]}

        lm = model.to_labelmodel(labelcompounds=labelcompounds, labelmaps=labelmaps)
        self.assertEqual(lm.label_compounds["X"]["num_labels"], 1)
        self.assertEqual(lm.label_compounds["X"]["isotopomers"], ["X__0", "X__1"])
        self.assertEqual(lm.label_compounds["Y"]["num_labels"], 1)
        self.assertEqual(lm.label_compounds["Y"]["isotopomers"], ["Y__0", "Y__1"])
        self.assertEqual(lm.nonlabel_compounds, ["ATP", "ADP"])
        rate = lm.rates["v1__0"]
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["X__0", "ATP"])
        self.assertEqual(rate["products"], ["Y__0", "ADP"])
        self.assertEqual(rate["modifiers"], ["Z__total"])
        self.assertEqual(rate["dynamic_variables"], ["X__0", "ATP", "Z__total"])
        self.assertEqual(rate["reversible"], False)
        rate = lm.rates["v1__1"]
        self.assertEqual(rate["parameters"], ["k1"])
        self.assertEqual(rate["substrates"], ["X__1", "ATP"])
        self.assertEqual(rate["products"], ["Y__1", "ADP"])
        self.assertEqual(rate["modifiers"], ["Z__total"])
        self.assertEqual(rate["dynamic_variables"], ["X__1", "ATP", "Z__total"])
        self.assertEqual(rate["reversible"], False)

    def test_to_labelmodel_reaction_not_in_labelmaps(self):
        model = Model()
        model.add_compounds(("x", "y", "z"))
        model.add_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            modifiers=["z"],
        )
        labelcompounds = {}
        labelmaps = {}
        lm = model.to_labelmodel(labelcompounds=labelcompounds, labelmaps=labelmaps)
        self.assertEqual(lm.label_compounds, {})
        self.assertEqual(lm.nonlabel_compounds, ["x", "y", "z"])
        self.assertEqual(lm.compounds, ["x", "y", "z"])
        rate = lm.rates["v1"]
        self.assertEqual(rate["substrates"], ["x"])
        self.assertEqual(rate["products"], ["y"])
        self.assertEqual(rate["modifiers"], ["z"])

    def test_to_labelmodel_algebraic_module(self):
        model = Model()
        model.add_compounds(("x", "z"))
        model.add_algebraic_module(
            module_name="m1",
            function=lambda *args: 0,
            compounds=["x"],
            derived_compounds=["A"],
            modifiers=["z"],
        )
        labelcompounds = {"x": 2, "z": 2}
        labelmaps = {}
        lm = model.to_labelmodel(labelcompounds=labelcompounds, labelmaps=labelmaps)

        mod = lm.algebraic_modules["m1"]
        self.assertEqual(mod["compounds"], ["x__total"])
        self.assertEqual(mod["derived_compounds"], ["A"])
        self.assertEqual(mod["modifiers"], ["z__total"])

    def test_to_labelmodel_algebraic_module_not_in_labelcompounds(self):
        model = Model()
        model.add_compounds(("x", "z"))
        model.add_algebraic_module(
            module_name="m1",
            function=lambda *args: 0,
            compounds=["x"],
            derived_compounds=["A"],
            modifiers=["z"],
        )
        labelcompounds = {}
        labelmaps = {}
        lm = model.to_labelmodel(labelcompounds=labelcompounds, labelmaps=labelmaps)

        mod = lm.algebraic_modules["m1"]
        self.assertEqual(mod["compounds"], ["x"])
        self.assertEqual(mod["derived_compounds"], ["A"])
        self.assertEqual(mod["modifiers"], ["z"])

    def test_to_linear_labelmodel(self):
        model = Model()
        model.add_compounds(("X", "Y", "ATP", "ADP", "Z"))
        model.add_reaction(
            rate_name="v1",
            function=lambda x, ATP, z: x * ATP / z,
            stoichiometry={"X": -1, "ATP": -1, "Y": 1, "ADP": 1},
            modifiers=["Z"],
            parameters=["k1"],
        )

        labelcompounds = {"X": 1, "Y": 1, "Z": 1}
        labelmaps = {"v1": [0]}

        lm = model.to_linear_labelmodel(
            labelcompounds=labelcompounds, labelmaps=labelmaps
        )
        self.assertEqual(lm.compounds, ["X__0", "Y__0", "Z__0"])
        self.assertEqual(lm.stoichiometries, {"v1__0": {"X__0": -1, "Y__0": 1}})

        rate = lm.rates["v1__0"]
        self.assertEqual(rate["base_name"], "v1")
        self.assertEqual(rate["substrate"], "X__0")

    def test_to_linear_labelmodel_no_label_info(self):
        model = Model()
        model.add_compounds(("X", "Y", "ATP", "ADP", "Z"))
        model.add_reaction(
            rate_name="v1",
            function=lambda x, ATP, z: x * ATP / z,
            stoichiometry={"X": -1, "ATP": -1, "Y": 1, "ADP": 1},
            modifiers=["Z"],
            parameters=["k1"],
        )

        labelcompounds = {}
        labelmaps = {}

        with self.assertWarns(UserWarning):
            lm = model.to_linear_labelmodel(
                labelcompounds=labelcompounds, labelmaps=labelmaps
            )

        self.assertEqual(lm.compounds, [])
        self.assertEqual(lm.rates, {})
        self.assertEqual(lm.stoichiometries, {})

    def test_to_linear_labelmodel_warn_on_reversible(self):
        model = Model()
        model.add_compounds(("X", "Y", "ATP", "ADP", "Z"))
        model.add_reaction(
            rate_name="v1",
            function=lambda x, ATP, Y, ADP, z: x * ATP / z,
            stoichiometry={"X": -1, "ATP": -1, "Y": 1, "ADP": 1},
            modifiers=["Z"],
            parameters=["k1"],
            reversible=True,
        )

        labelcompounds = {"X": 1, "Y": 1, "Z": 1}
        labelmaps = {"v1": [0]}

        with self.assertWarns(UserWarning):
            lm = model.to_linear_labelmodel(
                labelcompounds=labelcompounds, labelmaps=labelmaps
            )
            self.assertEqual(lm.stoichiometries, {"v1__0": {"X__0": -1, "Y__0": 1}})
            self.assertEqual(lm.compounds, ["X__0", "Y__0", "Z__0"])
            rate = lm.rates["v1__0"]
            self.assertEqual(rate["base_name"], "v1")
            self.assertEqual(rate["substrate"], "X__0")


class SBMLTests(unittest.TestCase):
    def test_create_sbml_rates_without_function(self):
        model = Model()
        model.add_reaction(
            rate_name="v1",
            function=lambda x, y, ATP, ADP: x * ATP - y * ADP,
            stoichiometry={"x": -2, "y": 1},
            modifiers=["ATP", "ADP"],
            parameters=["k1"],
            reversible=True,
        )

        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_reactions(sbml_model=sbml_model)

        rxn = sbml_model.getReaction("v1")
        self.assertEqual(rxn.getId(), "v1")
        self.assertEqual(rxn.getReversible(), True)
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "x")
        self.assertEqual(rxn.getListOfReactants()[0].getStoichiometry(), 2.0)
        self.assertEqual(rxn.getListOfReactants()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "y")
        self.assertEqual(rxn.getListOfProducts()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfProducts()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfModifiers()[0].getSpecies(), "ATP")
        self.assertEqual(rxn.getListOfModifiers()[1].getSpecies(), "ADP")
        self.assertEqual(rxn.getKineticLaw(), None)

    def test_create_sbml_rates_with_meta_info(self):
        model = Model()
        model.add_reaction(
            rate_name="v1",
            function=lambda x, y, ATP, ADP: x * ATP - y * ADP,
            stoichiometry={"x": -2, "y": 1},
            modifiers=["ATP", "ADP"],
            parameters=["k1"],
            reversible=True,
            **{"sbml_function": "x * ATP - y * ADP", "common_name": "reaction-one"},
        )

        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_reactions(sbml_model=sbml_model)

        rxn = sbml_model.getReaction("v1")
        self.assertEqual(rxn.getId(), "v1")
        self.assertEqual(rxn.getName(), "reaction-one")
        self.assertEqual(rxn.getReversible(), True)
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "x")
        self.assertEqual(rxn.getListOfReactants()[0].getStoichiometry(), 2.0)
        self.assertEqual(rxn.getListOfReactants()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "y")
        self.assertEqual(rxn.getListOfProducts()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfProducts()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfModifiers()[0].getSpecies(), "ATP")
        self.assertEqual(rxn.getListOfModifiers()[1].getSpecies(), "ADP")
        self.assertEqual(rxn.getKineticLaw().getFormula(), "x * ATP - y * ADP")

    def test_warn_algebraic_modules(self):
        model = Model()
        model.add_algebraic_module(
            module_name="mod1",
            function=lambda *args: 0,
            compounds=["x", "y"],
            derived_compounds=["A1"],
            modifiers=["z"],
            parameters=["p1"],
        )
        with self.assertWarns(UserWarning):
            model._model_to_sbml()


# New tests
def test_add_reaction_from_args():
    parameters = {"k": 1}
    model = Model(parameters=parameters)
    model.add_compounds(["x", "y", "xi"])
    model.add_reaction_from_args(
        rate_name="v1",
        function=lambda x, y, time, k: k / time * (x - y),
        stoichiometry={"x": -1, "y": 1},
        args=["x", "y", "time", "k"],
        reversible=True,
    )
    rate = model.rates["v1"]
    assert rate["parameters"] == ["k"]
    assert rate["substrates"] == ["x"]
    assert rate["products"] == ["y"]
    assert rate["modifiers"] == ["time"]
    assert rate["dynamic_variables"] == ["x", "y", "time"]
    assert rate["reversible"]
    assert model.stoichiometries == {"v1": {"x": -1, "y": 1}}
    assert model.stoichiometries_by_compounds == {"x": {"v1": -1}, "y": {"v1": 1}}
