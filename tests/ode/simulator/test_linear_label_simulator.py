# type: ignore

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from modelbase2.ode import LinearLabelModel, Model, Simulator
from modelbase2.ode import ratefunctions as rf


def BUILD_MODEL():
    p = {
        "kf_TPI": 1.0,
        "Keq_TPI": 21.0,
        "kf_Ald": 2000.0,
        "Keq_Ald": 7000.0,
    }

    p["kr_TPI"] = p["kf_TPI"] / p["Keq_TPI"]
    p["kr_Ald"] = p["kf_Ald"] / p["Keq_Ald"]

    # Generate initial concentrations
    GAP0 = 2.5e-5
    DHAP0 = GAP0 * p["Keq_TPI"]
    FBP0 = GAP0 * DHAP0 * p["Keq_Ald"]

    base_y0 = {"GAP": GAP0, "DHAP": DHAP0, "FBP": FBP0}

    model = Model(p)
    model.add_compounds(["GAP", "DHAP", "FBP"])
    model.add_reaction(
        rate_name="TPIf",
        function=rf.mass_action_1,
        stoichiometry={"GAP": -1, "DHAP": 1},
        parameters=["kf_TPI"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="TPIr",
        function=rf.mass_action_1,
        stoichiometry={"DHAP": -1, "GAP": 1},
        parameters=["kr_TPI"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="ALDf",
        function=rf.mass_action_2,
        stoichiometry={"DHAP": -1, "GAP": -1, "FBP": 1},
        parameters=["kf_Ald"],
        reversible=False,
    )
    model.add_reaction(
        rate_name="ALDr",
        function=rf.mass_action_1,
        stoichiometry={"FBP": -1, "DHAP": 1, "GAP": 1},
        parameters=["kr_Ald"],
        reversible=False,
    )

    y_ss = base_y0
    v_ss = {k: v[0] for k, v in model.get_fluxes(y_ss).items()}

    llm = model.to_linear_labelmodel(
        labelcompounds={"GAP": 3, "DHAP": 3, "FBP": 6},
        labelmaps={
            "TPIf": [2, 1, 0],
            "TPIr": [2, 1, 0],
            "ALDf": [0, 1, 2, 3, 4, 5],
            "ALDr": [0, 1, 2, 3, 4, 5],
        },
    )
    y0 = llm.generate_y0(initial_labels={"GAP": 0})
    return llm, y0, y_ss, v_ss


def SIMULATE_MODEL():
    llm, y0, y_ss, v_ss = BUILD_MODEL()
    s = Simulator(model=llm)
    s.initialise(label_y0=y0, y_ss=y_ss, v_ss=v_ss, external_label=1)
    t, y = s.simulate(time_points=[0, 10_000])
    return s


SIM = SIMULATE_MODEL()


class SimulationTests(unittest.TestCase):
    def test_initialise(self):
        llm, y0, y_ss, v_ss = BUILD_MODEL()
        s = Simulator(llm)
        s.initialise(label_y0=y0, y_ss=y_ss, v_ss=v_ss, external_label=1)
        self.assertEqual(s.model._y_ss, y_ss)
        self.assertEqual(s.model._v_ss, v_ss)
        self.assertEqual(s.model._external_label, 1)

    def test_copy(self):
        llm, y0, y_ss, v_ss = BUILD_MODEL()
        s1 = Simulator(model=llm)
        s1.initialise(label_y0=y0, y_ss=y_ss, v_ss=v_ss, external_label=1)

        s2 = s1.copy()
        self.assertEqual(s2.time, None)
        self.assertEqual(s2.results, None)
        self.assertIsNot(s1, s2)

    def test_copy_nonempty(self):
        llm, y0, y_ss, v_ss = BUILD_MODEL()
        s1 = Simulator(model=llm)
        s1.initialise(label_y0=y0, y_ss=y_ss, v_ss=v_ss, external_label=1)
        t, y = s1.simulate(time_points=[0, 10_000])

        s2 = s1.copy()
        np.testing.assert_array_equal(s2.time, s1.time)
        np.testing.assert_array_equal(s2.results, s1.results)
        self.assertIsNot(s1, s2)

    def test_simulate(self):
        llm, y0, y_ss, v_ss = BUILD_MODEL()
        y0 = llm.generate_y0(initial_labels={"GAP": 0})
        lls = Simulator(llm)
        lls.initialise(y0, y_ss, v_ss)
        t, y = lls.simulate(20)

    def test_get_label_position(self):
        llm, y0, y_ss, v_ss = BUILD_MODEL()
        lls = SIM.copy()
        np.testing.assert_array_equal(
            lls.get_label_position(compound="GAP", position=1), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="GAP", position=2), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="DHAP", position=0), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="DHAP", position=1), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="FBP", position=0), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="FBP", position=1), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="FBP", position=4), [0, 0]
        )
        np.testing.assert_array_equal(
            lls.get_label_position(compound="FBP", position=5), [0, 0]
        )

        np.testing.assert_array_almost_equal(
            (
                lls.get_label_position(compound="GAP", position=0)
                + lls.get_label_position(compound="DHAP", position=2)
                * y_ss["DHAP"]
                / y_ss["GAP"]
                + lls.get_label_position(compound="FBP", position=2)
                * y_ss["FBP"]
                / y_ss["GAP"]
                + lls.get_label_position(compound="FBP", position=3)
                * y_ss["FBP"]
                / y_ss["GAP"]
            ),
            [1, 1],
        )
        np.testing.assert_array_almost_equal(
            (
                lls.get_label_distribution(compound="GAP")[:, 0]
                + lls.get_label_distribution(compound="DHAP")[:, 2]
                * y_ss["DHAP"]
                / y_ss["GAP"]
                + lls.get_label_distribution(compound="FBP")[:, 2]
                * y_ss["FBP"]
                / y_ss["GAP"]
                + lls.get_label_distribution(compound="FBP")[:, 3]
                * y_ss["FBP"]
                / y_ss["GAP"]
            ),
            [1, 1],
        )

    def test_get_label_distribution(self):
        llm, y0, y_ss, v_ss = BUILD_MODEL()
        lls = SIM.copy()
        self.assertEqual(lls.get_label_distribution(compound="GAP").shape, (2, 3))
        self.assertEqual(lls.get_label_distribution(compound="DHAP").shape, (2, 3))
        self.assertEqual(lls.get_label_distribution(compound="FBP").shape, (2, 6))


class PlottingTests(unittest.TestCase):
    def test_plot_label_distribution(self):
        s = SIM.copy()
        fig, ax = s.plot_label_distribution(compound="GAP")
        plt.close()

    def test_plot_label_distribution_grid(self):
        s = SIM.copy()
        fig, ax = s.plot_label_distribution_grid(compounds=["GAP", "DHAP", "FBP"])
        plt.close()

    def test_plot_all_label_distributions(self):
        s = SIM.copy()
        fig, ax = s.plot_all_label_distributions()
        plt.close()
