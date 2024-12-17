# type: ignore

from __future__ import annotations

import unittest
import warnings

import numpy as np
from modelbase.ode import Model, Simulator
from modelbase.ode import ratefunctions as rf

try:
    from modelbase.ode.integrators import Assimulo

    class SimulatorBaseTests(unittest.TestCase):
        def test_initialise(self):
            parameters = {"alpha": 1}
            model = Model(parameters=parameters)
            model.add_compound(compound="x")
            y0 = {"x": 1}
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=False)
            self.assertEqual(s.y0, [1])
            self.assertEqual(s.integrator.__class__.__name__, "_IntegratorAssimulo")

        def test_get_kwargs(self):
            parameters = {"alpha": 1}
            model = Model(parameters=parameters)
            model.add_compound(compound="x")
            y0 = {"x": 1}
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=False)

            kwargs = s.get_integrator_params()
            self.assertEqual(kwargs["atol"][0], 1e-8)
            self.assertEqual(kwargs["backward"], False)
            self.assertEqual(kwargs["clock_step"], False)
            self.assertEqual(kwargs["discr"], "BDF")
            self.assertEqual(kwargs["display_progress"], True)
            self.assertEqual(kwargs["dqrhomax"], 0.0)
            self.assertEqual(kwargs["dqtype"], "CENTERED")
            self.assertEqual(kwargs["external_event_detection"], False)
            self.assertEqual(kwargs["inith"], 0.0)
            self.assertEqual(kwargs["linear_solver"], "DENSE")
            self.assertEqual(kwargs["maxcor"], 3)
            self.assertEqual(kwargs["maxcorS"], 3)
            self.assertEqual(kwargs["maxh"], 0.0)
            self.assertEqual(kwargs["maxkrylov"], 5)
            self.assertEqual(kwargs["maxncf"], 1)
            self.assertEqual(kwargs["maxnef"], 4)
            self.assertEqual(kwargs["maxord"], 5)
            self.assertEqual(kwargs["maxsteps"], 10000)
            self.assertEqual(kwargs["minh"], 0.0)
            self.assertEqual(kwargs["norm"], "WRMS")
            self.assertEqual(kwargs["num_threads"], 1)
            self.assertEqual(kwargs["pbar"], [])
            self.assertEqual(kwargs["precond"], "PREC_NONE")
            self.assertEqual(kwargs["report_continuously"], False)
            self.assertEqual(kwargs["rtol"], 1e-8)
            self.assertEqual(kwargs["sensmethod"], "STAGGERED")
            self.assertEqual(kwargs["suppress_sens"], False)
            self.assertEqual(kwargs["time_limit"], 0)
            self.assertEqual(kwargs["usejac"], False)
            self.assertEqual(kwargs["usesens"], False)
            self.assertEqual(kwargs["verbosity"], 50)

        def test_simulation_kwargs(self):
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
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(
                y0=y0,
                test_run=True,
            )
            t, y = s.simulate(time_points=[1, 2, 3], **{"atol": 1, "rtol": 1})
            self.assertEqual(s.integrator.integrator.atol, 1)
            self.assertEqual(s.integrator.integrator.rtol, 1)

        def test_simulate_to_steady_state_max_rounds(self):
            model = Model()
            model.add_compounds(compounds=["x", "y"])
            model.add_reaction(
                rate_name="v1", function=lambda x: x, stoichiometry={"x": -1, "y": 1}
            )
            y0 = {"x": 1, "y": 0}
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=False)
            t, y = s.simulate_to_steady_state(simulation_kwargs={"max_rounds": 1})
            np.testing.assert_array_almost_equal(y[0], [0, 1])

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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=False)
            t, y = s.simulate(time_points=np.arange(0, 11))
            self.assertTrue(np.isclose(y, np.exp(range(11)).reshape(-1, 1)).all())

        def test_simulate_to_steady_state(self):
            model = Model()
            model.add_compounds(compounds=["x", "y"])
            model.add_reaction(
                rate_name="v1", function=lambda x: x, stoichiometry={"x": -1, "y": 1}
            )
            y0 = {"x": 1, "y": 0}
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=False)
            t, y = s.simulate_to_steady_state()
            np.testing.assert_array_almost_equal(y[0], [0, 1])

        def test_simulate_to_steady_state_with_prior(self):
            model = Model()
            model.add_compounds(compounds=["x", "y"])
            model.add_reaction(
                rate_name="v1", function=lambda x: x, stoichiometry={"x": -1, "y": 1}
            )
            y0 = {"x": 1, "y": 0}
            s = Simulator(model=model, integrator=Assimulo)
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
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=True)
            t, y = s.simulate_to_steady_state()
            assert t is None
            assert y is None

        def test_parameter_scan_fail(self):
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
            s = Simulator(model=model, integrator=Assimulo)
            s.initialise(y0=y0, test_run=True)
            res = s.parameter_scan(
                parameter_name="kf",
                parameter_values=(1,),
                multiprocessing=False,
            )
            np.testing.assert_array_equal(res["x"], [np.nan])
            np.testing.assert_array_equal(res["y"], [np.nan])

    class IntegratorTests(unittest.TestCase):
        def test_init(self):
            pass

        def test_reset(self):
            pass

        def test_simulate(self):
            pass

        def test_simulate_t_end(self):
            pass

        def test_simulate_steps(self):
            pass

        def test_simulate_time_points(self):
            pass

        def test_simulate_steps_and_time_points(self):
            pass

        def test_simulate_kwargs(self):
            pass

        def test_simulate_to_steady_state(self):
            pass

        def test_simulate_to_steady_state_kwargs(self):
            pass

except ImportError:
    warnings.warn("Assmiulo not installed, skipping tests")
