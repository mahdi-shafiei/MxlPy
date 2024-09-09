# type: ignore

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt

from modelbase2.ode import Model, Simulator


def mass_action(s, kf):
    return kf * s


def reversible_mass_action(s, p, kf, kr):
    return (kf * s) - (kr * p)


def generate_simulation_results():
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
        function=mass_action,
        stoichiometry={"c1": -1, "c2": 1},
        modifiers=None,
        parameters=["kf"],
        reversible=False,
    )
    # Reversible reaction
    model.add_reaction(
        rate_name="reversible_reaction",
        function=reversible_mass_action,
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
    s = Simulator(model)
    s.initialise(y0=y0)
    t, y = s.simulate(t_end=10, steps=10)
    return s.copy()


s = generate_simulation_results()
model = s.model


class PlottingTests(unittest.TestCase):
    def test_plot_selection(self):
        fig, ax = s.plot_selection(compounds=["a", "b"])
        plt.close()

    def test_plot(self):
        fig, ax = s.plot()
        plt.close()

    def test_plot_log(self):
        fig, ax = s.plot_log()
        plt.close()

    def test_plot_semilog_x(self):
        fig, ax = s.plot_semilog(log_axis="x")
        plt.close()

    def test_plot_semilog_y(self):
        fig, ax = s.plot_semilog(log_axis="y")
        plt.close()

    def test_plot_semilog_fail(self):
        with self.assertRaises(ValueError):
            fig, ax = s.plot_semilog(log_axis="z")

    def test_plot_existing_axes(self):
        fig, ax = plt.subplots(1, 1)
        fig, ax = s.plot(ax=ax)
        plt.close()

    def test_plot_derived(self):
        fig, ax = s.plot_derived()
        plt.close()

    def test_plot_all(self):
        fig, ax = s.plot_all()
        plt.close()

    def test_plot_grid_1_2(self):
        fig, ax = s.plot_grid(compound_groups=[["a"], ["b"]], ncols=2)
        plt.close()

    def test_plot_grid_1_2_not_shared(self):
        fig, ax = s.plot_grid(
            compound_groups=[["a"], ["b"]],
            ncols=2,
            sharex=False,
            sharey=False,
        )
        plt.close()

    def test_plot_grid_2_1(self):
        fig, ax = s.plot_grid(compound_groups=[["a"], ["b"]], ncols=1)
        plt.close()

    def test_plot_grid_2_1_not_shared(self):
        fig, ax = s.plot_grid(
            compound_groups=[["a"], ["b"]],
            ncols=1,
            sharex=False,
            sharey=False,
        )
        plt.close()

    def test_plot_grid_2_2(self):
        fig, ax = s.plot_grid(compound_groups=[["a"], ["b"], ["c1"], ["c2"]], ncols=2)
        plt.close()

    def test_plot_grid_2_2_not_shared(self):
        fig, ax = s.plot_grid(
            compound_groups=[["a"], ["b"], ["c1"], ["c2"]],
            ncols=2,
            sharex=False,
            sharey=False,
        )
        plt.close()

    def test_plot_selection_against_variable(self):
        fig, ax = s.plot_selection_against_variable(
            compounds=["a", "b"], variable="rate_time"
        )
        plt.close()

    def test_plot_against_variable(self):
        fig, ax = s.plot_against_variable(variable="rate_time")
        plt.close()

    def test_plot_derived_against_variable(self):
        fig, ax = s.plot_derived_against_variable(variable="rate_time")
        plt.close()

    def test_plot_all_against_variable(self):
        fig, ax = s.plot_all_against_variable(variable="rate_time")
        plt.close()

    def test_plot_flux_selection(self):
        fig, ax = s.plot_flux_selection(rate_names=["vanilla_reaction"])
        plt.close()

    def test_plot_fluxes_grid(self):
        fig, ax = s.plot_fluxes_grid(
            rate_groups=[["vanilla_reaction"], ["vanilla_reaction"]]
        )
        plt.close()

    def test_plot_flux_selection_against_variable(self):
        fig, ax = s.plot_flux_selection_against_variable(
            rate_names=["vanilla_reaction"], variable="rate_time"
        )
        plt.close()

    def test_plot_fluxes_against_variable(self):
        fig, ax = s.plot_fluxes_against_variable(variable="rate_time")
        plt.close()

    def test_plot_phase_plane(self):
        fig, ax = s.plot_phase_plane(cpd1="c1", cpd2="c2")
        plt.close()

    def test_plot_phase_space(self):
        fig, ax = s.plot_phase_space(cpd1="c1", cpd2="c2", cpd3="rate_time")
        plt.close()

    def test_plot_phase_space_existing_axes(self):
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        fig, ax = s.plot_phase_space(cpd1="c1", cpd2="c2", cpd3="rate_time", ax=ax)
        plt.close()

    def test_plot_trajectories(self):
        fig, ax = s.plot_trajectories(
            y0=s.y0,
            t0=0,
            cpd1="c1",
            cpd2="c2",
            cpd1_bounds=(1, 4),
            cpd2_bounds=(1, 4),
            n=10,
        )
        plt.close()

    def test_plot_trajectories_existing_axes(self):
        fig, ax = plt.subplots(1, 1)
        fig, ax = s.plot_trajectories(
            y0=s.y0,
            t0=0,
            cpd1="c1",
            cpd2="c2",
            cpd1_bounds=(1, 4),
            cpd2_bounds=(1, 4),
            n=10,
            ax=ax,
        )
        plt.close()

    def test_plot_3d_trajectories(self):
        fig, ax = s.plot_3d_trajectories(
            y0=s.y0,
            t0=0,
            cpd1="c1",
            cpd2="c2",
            cpd3="rate_time",
            cpd1_bounds=(1, 4),
            cpd2_bounds=(1, 4),
            cpd3_bounds=(1, 4),
            n=10,
        )
        plt.close()

    def test_plot_3d_trajectories_existing_axes(self):
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        fig, ax = s.plot_3d_trajectories(
            y0=s.y0,
            t0=0,
            cpd1="c1",
            cpd2="c2",
            cpd3="rate_time",
            cpd1_bounds=(1, 4),
            cpd2_bounds=(1, 4),
            cpd3_bounds=(1, 4),
            n=10,
            ax=ax,
        )
        plt.close()
