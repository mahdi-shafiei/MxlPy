# type: ignore

from __future__ import annotations

import unittest

import numpy as np

from modelbase2.ode import Model, Simulator
from modelbase2.ode import ratefunctions as rf


def GENERATE_MODEL():
    parameters = {"k_in": 1, "kf": 1, "kr": 1}
    model = Model(parameters=parameters)
    model.add_compounds(["a", "b", "c1", "c2", "d1", "d2", "rate_time"])

    # Vanilla algebraic module
    model.add_algebraic_module(
        module_name="vanilla_module",
        function=lambda a: (a, 2 * a),
        compounds=["a"],
        derived_compounds=["a1", "a2"],
        modifiers=None,
        parameters=None,
    )

    # Time-dependent algebraic module
    model.add_algebraic_module(
        module_name="time_dependent_module",
        function=lambda time: (time,),
        compounds=None,
        derived_compounds=["derived_time"],
        modifiers=["time"],
        parameters=None,
    )

    # Singleton reaction
    model.add_reaction(
        rate_name="singleton_reaction",
        function=lambda k_in: k_in,
        stoichiometry={"b": 1},
        modifiers=None,
        parameters=["k_in"],
        reversible=False,
    )

    # Time-dependent reaction
    model.add_reaction(
        rate_name="time_dependent_reaction",
        function=lambda time: time,
        stoichiometry={"rate_time": 1},
        modifiers=["time"],
        parameters=None,
        reversible=False,
    )
    # Vanilla reaction
    model.add_reaction(
        rate_name="vanilla_reaction",
        function=rf.mass_action_1,
        stoichiometry={"c1": -1, "c2": 1},
        modifiers=None,
        parameters=["kf"],
        reversible=False,
    )
    # Reversible reaction
    model.add_reaction(
        rate_name="reversible_reaction",
        function=rf.reversible_mass_action_1_1,
        stoichiometry={"d1": -1, "d2": 1},
        modifiers=None,
        parameters=["kf", "kr"],
        reversible=True,
    )
    y0 = {
        "a": 1,
        "b": 1,
        "c1": 1,
        "c2": 0,
        "d1": 1,
        "d2": 0,
        "rate_time": 1,
    }
    return model, y0


def GENERATE_RESULTS(model, y0):
    s = Simulator(
        model,
    )
    s.initialise(y0=y0)
    t, y = s.simulate(t_end=10, steps=10)
    return s.copy()


MODEL, Y0 = GENERATE_MODEL()
SIM = GENERATE_RESULTS(model=MODEL, y0=Y0)


class SimulatorBaseTests(unittest.TestCase):
    def test_init(self):
        model = Model()
        s = Simulator(model=model)
        self.assertEqual(s.time, None)
        self.assertEqual(s.results, None)
        self.assertEqual(s.full_results, None)

    def test_update_parameters(self):
        model = Model(parameters={"k1": 1})
        s = Simulator(model=model)
        s.update_parameters(parameters={"k1": 2})
        self.assertEqual(s.model.parameters["k1"], 2)

    def test_copy(self):
        model = Model(parameters={"k_in": 1})
        model.add_compound(compound="x")
        model.add_reaction(
            rate_name="v1",
            function=rf.constant,
            stoichiometry={"x": 1},
            parameters=["k_in"],
        )
        s1 = Simulator(model=model)
        s1.initialise({"x": 0})

        s2 = s1.copy()
        self.assertEqual(s2.time, None)
        self.assertEqual(s2.results, None)
        self.assertIsNot(s1, s2)

    def test_copy_nonempty(self):
        model = Model(parameters={"k_in": 1})
        model.add_compound(compound="x")
        model.add_reaction(
            rate_name="v1",
            function=rf.constant,
            stoichiometry={"x": 1},
            parameters=["k_in"],
        )
        s1 = Simulator(model=model)
        s1.initialise({"x": 0})
        t, y = s1.simulate(1)

        s2 = s1.copy()
        np.testing.assert_array_equal(s2.time, s1.time)
        np.testing.assert_array_equal(s2.results, s1.results)
        self.assertIsNot(s1, s2)


class SimulationTests(unittest.TestCase):
    def test_simulate(self):
        parameters = {"alpha": 3}
        model = Model(parameters=parameters)
        model.add_derived_parameter(
            parameter_name="beta",
            function=lambda alpha: 3 - alpha,
            parameters=["alpha"],
        )
        model.add_compound(compound="x")
        model.add_reaction(
            rate_name="v1",
            function=lambda x, alpha: alpha * x,
            stoichiometry={"x": 1},
            modifiers=["x"],
            parameters=["beta"],
            reversible=False,
        )
        y0 = {"x": 1}
        s = Simulator(
            model=model,
        )
        s.update_parameter(parameter_name="alpha", parameter_value=2)
        s.initialise(
            y0=y0,
            test_run=False,
        )
        t, y = s.simulate(t_end=10)
        self.assertTrue(np.isclose(y[-1][0], np.exp(10)))
        self.assertEqual(s.full_results, None)

    def test_simulate_to_steady_state(self):
        model = Model(parameters={"kf_base": 3})
        model.add_derived_parameter(
            parameter_name="kf",
            function=lambda kf_base: 3 - kf_base,
            parameters=["kf_base"],
        )
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=lambda x, kf: kf * x,
            stoichiometry={"x": -1, "y": 1},
            parameters=["kf"],
        )
        y0 = {"x": 1, "y": 0}
        s = Simulator(
            model=model,
        )
        s.update_parameter(parameter_name="kf_base", parameter_value=2)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate_to_steady_state()
        np.testing.assert_array_almost_equal(y[0], [0, 1])
        self.assertEqual(s.full_results, None)

    def test_parameter_scan(self):
        parameters = {"kf": 1, "kr": 1}
        model = Model(parameters=parameters)
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=rf.reversible_mass_action_1_1,
            stoichiometry={"x": -1, "y": 1},
            parameters=["kf", "kr"],
            reversible=True,
        )
        y0 = {"x": 1, "y": 0}
        s = Simulator(
            model=model,
        )
        s.initialise(y0=y0, test_run=True)
        res = s.parameter_scan(
            parameter_name="kf",
            parameter_values=(1, 3, 4, 7, 9, 15),
            tolerance=1e-6,
            multiprocessing=False,
        )
        np.testing.assert_allclose(
            res["x"], [0.5, 0.25, 0.2, 0.125, 0.1, 0.0625], rtol=1e-6
        )
        np.testing.assert_allclose(
            res["y"], [0.5, 0.75, 0.8, 0.875, 0.9, 0.9375], rtol=1e-6
        )

    def test_parameter_scan_multiprocessing(self):
        parameters = {"kf": 1, "kr": 1}
        model = Model(parameters=parameters)
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=rf.reversible_mass_action_1_1,
            stoichiometry={"x": -1, "y": 1},
            parameters=["kf", "kr"],
            reversible=True,
        )
        y0 = {"x": 1, "y": 0}
        s = Simulator(
            model=model,
        )
        s.initialise(y0=y0, test_run=True)
        res = s.parameter_scan(
            parameter_name="kf",
            parameter_values=(1, 3, 4, 7, 9, 15),
            tolerance=1e-6,
            multiprocessing=True,
        )

        np.testing.assert_allclose(
            res["x"], [0.5, 0.25, 0.2, 0.125, 0.1, 0.0625], rtol=1e-6
        )
        np.testing.assert_allclose(
            res["y"], [0.5, 0.75, 0.8, 0.875, 0.9, 0.9375], rtol=1e-6
        )

    def test_get_full_results_array_shape(self):
        s = SIM.copy()
        self.assertEqual(s.get_full_results_array(concatenated=True).shape, (11, 10))
        self.assertEqual(
            s.get_full_results_array(concatenated=False)[0].shape, (11, 10)
        )
        self.assertEqual(s.full_results[0].shape, (11, 10))

    def test_get_full_results_dict_shape(self):
        s = SIM.copy()
        res = s.get_full_results_dict(concatenated=True)
        self.assertEqual(res["a"].shape, (11,))
        self.assertEqual(res["b"].shape, (11,))
        self.assertEqual(res["c1"].shape, (11,))
        self.assertEqual(res["c2"].shape, (11,))
        self.assertEqual(res["d1"].shape, (11,))
        self.assertEqual(res["d2"].shape, (11,))
        self.assertEqual(res["rate_time"].shape, (11,))
        self.assertEqual(res["a1"].shape, (11,))
        self.assertEqual(res["a2"].shape, (11,))
        self.assertEqual(res["derived_time"].shape, (11,))

        res = s.get_full_results_dict(concatenated=False)[0]
        self.assertEqual(res["a"].shape, (11,))
        self.assertEqual(res["b"].shape, (11,))
        self.assertEqual(res["c1"].shape, (11,))
        self.assertEqual(res["c2"].shape, (11,))
        self.assertEqual(res["d1"].shape, (11,))
        self.assertEqual(res["d2"].shape, (11,))
        self.assertEqual(res["rate_time"].shape, (11,))
        self.assertEqual(res["a1"].shape, (11,))
        self.assertEqual(res["a2"].shape, (11,))
        self.assertEqual(res["derived_time"].shape, (11,))

    def test_get_full_results_df_shape(self):
        s = SIM.copy()
        res = s.get_full_results_df(concatenated=True)
        self.assertEqual(res.values.shape, (11, 10))
        self.assertEqual(
            res.index.values.tolist(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        self.assertEqual(
            res.columns.values.tolist(),
            ["a", "b", "c1", "c2", "d1", "d2", "rate_time", "a1", "a2", "derived_time"],
        )
        res = s.get_full_results_df(concatenated=False)[0]
        self.assertEqual(res.values.shape, (11, 10))
        self.assertEqual(
            res.index.values.tolist(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        self.assertEqual(
            res.columns.values.tolist(),
            ["a", "b", "c1", "c2", "d1", "d2", "rate_time", "a1", "a2", "derived_time"],
        )

    def test_get_variable_shape(self):
        s = SIM.copy()
        self.assertEqual(s.get_variable(variable="a", concatenated=True).shape, (11,))
        self.assertEqual(
            s.get_variable(variable="a", concatenated=False)[0].shape, (11,)
        )

    def test_get_variables_shape(self):
        s = SIM.copy()
        y = s.get_variables(variables=["a", "b"], concatenated=True)
        self.assertEqual(y.shape, (11, 2))

        y = s.get_variables(variables=["a", "b"], concatenated=False)[0]
        self.assertEqual(y.shape, (11, 2))

    def test_parameter_change(self):
        def alg_mod(module_par):
            return (module_par,)

        def rate_func(A):
            return A

        model = Model()
        model.add_compound("x")
        model.add_parameter("mod_par", 1)
        model.add_algebraic_module(
            module_name="mod_1",
            function=alg_mod,
            derived_compounds=["A"],
            parameters=["mod_par"],
        )
        model.add_reaction(
            rate_name="rate_1",
            function=rate_func,
            stoichiometry={"x": 1},
            modifiers=["A"],
        )

        s = Simulator(model=model)
        s.initialise({"x": 0})
        t, y = s.simulate(t_end=1, time_points=[1])
        s.update_parameter("mod_par", 0.5)
        t, y = s.simulate(t_end=2, time_points=[2])

        np.testing.assert_array_almost_equal(s.get_time(), [0.0, 1.0, 2.0])
        #
        t = s.get_time(concatenated=False)
        np.testing.assert_array_almost_equal(t[0], [0.0, 1.0])
        np.testing.assert_array_almost_equal(t[1], [2.0])
        #
        np.testing.assert_array_almost_equal(
            s.get_results_array(concatenated=True), [[0.0], [1.0], [1.5]]
        )
        #
        r = s.get_results_array(concatenated=False)
        np.testing.assert_array_almost_equal(r[0], [[0.0], [1.0]])
        np.testing.assert_array_almost_equal(r[1], [[1.5]])
        #
        d = s.get_results_dict()
        self.assertEqual(list(d.keys()), ["x"])
        np.testing.assert_array_almost_equal(d["x"], [0.0, 1.0, 1.5])
        #
        df = s.get_results()
        self.assertEqual(list(df.keys()), ["x"])
        self.assertEqual(list(df.index), [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(df.values, [[0.0], [1.0], [1.5]])
        #
        np.testing.assert_array_almost_equal(
            s.get_full_results_array(), np.array([[0.0, 1.0], [1.0, 1.0], [1.5, 0.5]])
        )
        #
        d = s.get_full_results_dict()
        self.assertEqual(list(d.keys()), ["x", "A"])
        np.testing.assert_array_almost_equal(d["x"], [0.0, 1.0, 1.5])
        np.testing.assert_array_almost_equal(d["A"], [1.0, 1.0, 0.5])
        #
        df = s.get_full_results()
        self.assertEqual(list(df.keys()), ["x", "A"])
        self.assertEqual(list(df.index), [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(
            df.values, [[0.0, 1.0], [1.0, 1.0], [1.5, 0.5]]
        )
        #
        np.testing.assert_array_almost_equal(
            s.get_fluxes_array(), [[1.0], [1.0], [0.5]]
        )
        #
        d = s.get_fluxes_dict()
        self.assertEqual(list(d.keys()), ["rate_1"])
        np.testing.assert_array_almost_equal(d["rate_1"], np.array([1.0, 1.0, 0.5]))
        #
        df = s.get_fluxes()
        self.assertEqual(list(df.keys()), ["rate_1"])
        self.assertEqual(list(df.index), [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(d["rate_1"], np.array([1.0, 1.0, 0.5]))

    def test_normalise_results(self):
        parameters = {"alpha": 1}
        model = Model(parameters=parameters)
        model.add_compound(compound="x")
        model.add_reaction(
            rate_name="v1",
            function=lambda x, alpha: alpha * x,
            stoichiometry={"x": 1},
            modifiers=["x"],
            parameters=["alpha"],
            reversible=False,
        )
        y0 = {"x": 1}
        s = Simulator(model=model)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=10)

        np.testing.assert_array_equal(y, s.get_results_array())
        np.testing.assert_array_equal(y, s.get_results_array(normalise=1))

        # array
        np.testing.assert_array_equal(y / 2, s.get_results_array(normalise=2))
        np.testing.assert_array_equal(
            y / 2, s.get_results_array(normalise=np.ones(len(y)) * 2)
        )
        np.testing.assert_array_equal(
            y / 2, s.get_results_array(normalise=[np.ones(len(y)) * 2])
        )

        # full array
        np.testing.assert_array_equal(y / 2, s.get_full_results_array(normalise=2))
        np.testing.assert_array_equal(
            y / 2, s.get_full_results_array(normalise=np.ones(len(y)) * 2)
        )
        np.testing.assert_array_equal(
            y / 2, s.get_full_results_array(normalise=[np.ones(len(y)) * 2])
        )

        # dict
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results_dict(normalise=2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results_dict(normalise=np.ones(len(y)) * 2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results_dict(normalise=[np.ones(len(y)) * 2])["x"]
        )

        # dict
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results_dict(normalise=2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results_dict(normalise=np.ones(len(y)) * 2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results_dict(normalise=[np.ones(len(y)) * 2])["x"]
        )

        # full dict
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_full_results_dict(normalise=2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(),
            s.get_full_results_dict(normalise=np.ones(len(y)) * 2)["x"],
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(),
            s.get_full_results_dict(normalise=[np.ones(len(y)) * 2])["x"],
        )

        # df
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results(normalise=2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results(normalise=np.ones(len(y)) * 2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_results(normalise=[np.ones(len(y)) * 2])["x"]
        )

        # full df
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_full_results(normalise=2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_full_results(normalise=np.ones(len(y)) * 2)["x"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(),
            s.get_full_results(normalise=[np.ones(len(y)) * 2])["x"],
        )

        # variable
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_variable(variable="x", normalise=2)
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(),
            s.get_variable(variable="x", normalise=np.ones(len(y)) * 2),
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(),
            s.get_variable(variable="x", normalise=[np.ones(len(y)) * 2]),
        )

        # variables
        np.testing.assert_array_equal(
            y / 2, s.get_variables(variables=["x"], normalise=2)
        )
        np.testing.assert_array_equal(
            y / 2, s.get_variables(variables=["x"], normalise=np.ones(len(y)) * 2)
        )
        np.testing.assert_array_equal(
            y / 2, s.get_variables(variables=["x"], normalise=[np.ones(len(y)) * 2])
        )

        # fluxes array
        np.testing.assert_array_equal(y / 2, s.get_fluxes_array(normalise=2))
        np.testing.assert_array_equal(
            y / 2, s.get_fluxes_array(normalise=np.ones(len(y)) * 2)
        )
        np.testing.assert_array_equal(
            y / 2, s.get_fluxes_array(normalise=[np.ones(len(y)) * 2])
        )

        # fluxes dict
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_fluxes_dict(normalise=2)["v1"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_fluxes_dict(normalise=np.ones(len(y)) * 2)["v1"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_fluxes_dict(normalise=[np.ones(len(y)) * 2])["v1"]
        )

        # fluxes df
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_fluxes(normalise=2)["v1"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_fluxes(normalise=np.ones(len(y)) * 2)["v1"]
        )
        np.testing.assert_array_equal(
            (y / 2).flatten(), s.get_fluxes(normalise=[np.ones(len(y)) * 2])["v1"]
        )
