# type: ignore

from __future__ import annotations

import unittest

import numpy as np

from modelbase2.ode import Model, Simulator
from modelbase2.ode import ratefunctions as rf
from modelbase2.ode.integrators import Scipy


class SimulatorBaseTests(unittest.TestCase):
    def test_initialise(self):
        parameters = {"alpha": 1}
        model = Model(parameters=parameters)
        model.add_compound(compound="x")
        y0 = {"x": 1}
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        self.assertEqual(s.y0, [1])
        self.assertEqual(s.integrator.__class__.__name__, "_IntegratorScipy")

    def test_get_kwargs(self):
        parameters = {"alpha": 1}
        model = Model(parameters=parameters)
        model.add_compound(compound="x")
        y0 = {"x": 1}
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)

        kwargs = s.get_integrator_params()
        kwargs_simulate = kwargs["simulate"]
        self.assertEqual(kwargs_simulate["ml"], None)
        self.assertEqual(kwargs_simulate["mu"], None)
        self.assertEqual(kwargs_simulate["rtol"], 1e-8)
        self.assertEqual(kwargs_simulate["atol"], 1e-8)
        self.assertEqual(kwargs_simulate["tcrit"], None)
        self.assertEqual(kwargs_simulate["h0"], 0.0)
        self.assertEqual(kwargs_simulate["hmax"], 0.0)
        self.assertEqual(kwargs_simulate["hmin"], 0.0)
        self.assertEqual(kwargs_simulate["ixpr"], 0)
        self.assertEqual(kwargs_simulate["mxstep"], 0)
        self.assertEqual(kwargs_simulate["mxhnil"], 0)
        self.assertEqual(kwargs_simulate["mxordn"], 12)
        self.assertEqual(kwargs_simulate["mxords"], 5)
        self.assertEqual(kwargs_simulate["printmessg"], 0)
        self.assertEqual(kwargs_simulate["tfirst"], False)

        kwargs_stss = kwargs["simulate_to_steady_state"]
        self.assertEqual(kwargs_stss["max_steps"], 100000)
        self.assertEqual(kwargs_stss["step_size"], 1)
        self.assertEqual(kwargs_stss["first_step"], None)
        self.assertEqual(kwargs_stss["min_step"], 0.0)
        self.assertEqual(kwargs_stss["max_step"], np.inf)
        self.assertEqual(kwargs_stss["rtol"], 1e-8)
        self.assertEqual(kwargs_stss["atol"], 1e-8)
        self.assertEqual(kwargs_stss["jac"], None)
        self.assertEqual(kwargs_stss["lband"], None)
        self.assertEqual(kwargs_stss["uband"], None)


class SimulationTests(unittest.TestCase):
    def test_simulation_steps_and_time_points(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=True)
        with self.assertWarns(UserWarning):
            t, y = s.simulate(steps=10, time_points=[1, 2, 3])
        np.testing.assert_array_equal(t, [0, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            y, np.exp([0, 1, 2, 3]).reshape(-1, 1), decimal=4
        )

    def test_simulation_one_variable_time_steps(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=10, steps=10)
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_time_steps_fail_without_y0(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        with self.assertRaises(ValueError):
            t, y = s.simulate(steps=10)

    def test_simulation_one_variable_time_points_range(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=10, time_points=range(11))
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_time_points_range_without_t_end(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(time_points=range(11))
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_time_points_list(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=10, time_points=list(range(11)))
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_time_points_list_without_t_end(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(time_points=list(range(11)))
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_time_points_array(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=10, time_points=np.arange(0, 11))
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_time_points_array_without_t_end(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(time_points=np.arange(0, 11))
        self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

    def test_simulation_one_variable_only_tend(self):
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
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=11)
        self.assertTrue(np.isclose(y[-1], np.exp(11)))

    def test_simulate_to_steady_state(self):
        model = Model()
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1", function=lambda x: x, stoichiometry={"x": -1, "y": 1}
        )
        y0 = {"x": 1, "y": 0}
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate_to_steady_state()
        np.testing.assert_array_almost_equal(y[0], [0, 1])

    def test_simulate_to_steady_state_kwargs(self):
        model = Model()
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1", function=lambda x: x, stoichiometry={"x": -1, "y": 1}
        )
        y0 = {"x": 1, "y": 0}
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate_to_steady_state(
            simulation_kwargs={
                "step_size": 1,
                "max_steps": 100_000,
                "integrator": "lsoda",
            }
        )
        np.testing.assert_array_almost_equal(y[0], [0, 1])

    def test_simulate_to_steady_state_with_prior(self):
        model = Model()
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1", function=lambda x: x, stoichiometry={"x": -1, "y": 1}
        )
        y0 = {"x": 1, "y": 0}
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=False)
        t, y = s.simulate(t_end=100, steps=10)
        self.assertEqual(s.time[0].shape, (11,))
        self.assertEqual(s.get_time().shape, (11,))
        self.assertEqual(s.results[0].shape, (11, 2))
        self.assertEqual(s.get_results_array().shape, (11, 2))
        t, y = s.simulate_to_steady_state()
        np.testing.assert_array_almost_equal(y[0], [0, 1])
        self.assertEqual(s.time[1].shape, (1,))
        self.assertEqual(s.get_time().shape, (12,))
        self.assertEqual(s.results[1].shape, (1, 2))
        self.assertEqual(s.get_results_array().shape, (12, 2))

    def test_simulate_to_steady_state_fail(self):
        parameters = {"kf": 1}
        model = Model(parameters=parameters)
        model.add_compounds(compounds=["x", "y"])
        model.add_reaction(
            rate_name="v1",
            function=rf.constant,
            stoichiometry={"x": 1, "y": 1},
            parameters=["kf"],
            reversible=False,
        )
        y0 = {"x": 0, "y": 0}
        s = Simulator(model=model, integrator=Scipy)
        s.initialise(y0=y0, test_run=True)
        t, y = s.simulate_to_steady_state()
        assert t is None
        assert y is None
