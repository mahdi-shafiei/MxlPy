from __future__ import annotations

import pandas as pd

from modelbase2 import Model, fns, mca

# def create_model() -> None:
#     parameters = {
#         "k_in": 1,
#         "k_fwd": 1,
#         "k_out": 1,
#     }

#     model = Model(parameters=parameters)
#     model.add_variables(compounds=["X", "Y"])
#     model.add_reaction(
#         rate_name="v_in",
#         function=rf.constant,
#         stoichiometry={"X": 1},
#         parameters=["k_in"],
#     )
#     model.add_reaction(
#         rate_name="v1",
#         function=rf.mass_action_1,
#         stoichiometry={"X": -1, "Y": 1},
#         parameters=["k_fwd"],
#     )
#     model.add_reaction(
#         rate_name="v_out",
#         function=rf.mass_action_1,
#         stoichiometry={"Y": -1},
#         parameters=["k_out"],
#     )
#     y0 = {"X": 1, "Y": 1}
#     parameters = ["k_in", "k_fwd", "k_out"]
#     compounds = ["X", "Y"]
#     return model, y0, parameters, compounds


def test_substrate_elasticitiy_mass_action() -> None:
    """Should yield 1 regardless of the concentration"""

    model = (
        Model()
        .add_parameters({"kf": 1})
        .add_variables({"x": 1.0, "y": 1.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            stoichiometry={"x": -1, "y": 1},
            args=["x", "kf"],
        )
    )
    pd.testing.assert_series_equal(
        mca.variable_elasticities(model=model, concs={"x": 1, "y": 0})["x"],
        pd.Series({"v1": 1.0}, name="x"),
        rtol=1e-6,
    )
    pd.testing.assert_series_equal(
        mca.variable_elasticities(model=model, concs={"x": 10, "y": 0})["x"],
        pd.Series({"v1": 1.0}, name="x"),
        rtol=1e-6,
    )
    pd.testing.assert_series_equal(
        mca.variable_elasticities(model=model, concs={"x": 100, "y": 0})["x"],
        pd.Series({"v1": 1.0}, name="x"),
        rtol=1e-6,
    )


def test_elasticities_michaelis_menten() -> None:
    model = (
        Model()
        .add_parameters({"vmax": 1, "km": 1})
        .add_variables({"x": 1.0, "y": 1.0})
        .add_reaction(
            "v1",
            fn=fns.michaelis_menten_1s,
            stoichiometry={"x": -1, "y": 1},
            args=["x", "vmax", "km"],
        )
    )
    # Should be 1 at x = 0
    pd.testing.assert_series_equal(
        mca.variable_elasticities(model=model, concs={"x": 1e-12, "y": 0})["x"],
        pd.Series({"v1": 1.0}, name="x"),
        rtol=1e-6,
    )
    # Should be 0 at x = km
    pd.testing.assert_series_equal(
        mca.variable_elasticities(model=model, concs={"x": 1, "y": 0})["x"],
        pd.Series({"v1": 0.5}, name="x"),
        rtol=1e-6,
    )
    # Should be 0 at x >> km
    pd.testing.assert_series_equal(
        mca.variable_elasticities(model=model, concs={"x": 1e12, "y": 0})["x"],
        pd.Series({"v1": 0.0}, name="x"),
        rtol=1e-6,
    )

    # Parameter elasticities
    pd.testing.assert_series_equal(
        mca.parameter_elasticities(model=model, parameters=["vmax"])["vmax"],
        pd.Series({"v1": 1.0}, name="vmax"),
        rtol=1e-6,
    )


# def test_get_compound_elasticity() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_compound_elasticity(
#             model=model, compound="X", y=y0, t=0, normalized=False
#         ),
#         [0.0, 1.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_compound_elasticity(
#             model=model, compound="X", y=y0, t=0, normalized=True
#         ),
#         [0.0, 1.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_compound_elasticity(
#             model=model, compound="Y", y=y0, t=0, normalized=False
#         ),
#         [0.0, 0.0, 1.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_compound_elasticity(
#             model=model, compound="Y", y=y0, t=0, normalized=True
#         ),
#         [0.0, 0.0, 1.0],
#         decimal=4,
#     )


# def test_get_compound_elasticities_array() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_compound_elasticities_array(
#             model=model,
#             compounds=compounds,
#             y=y0,
#             t=0,
#             normalized=False,
#         ),
#         [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_compound_elasticities_array_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_compound_elasticities_array(
#             model=model,
#             compounds=compounds,
#             y=y0,
#             t=0,
#             normalized=True,
#         ),
#         [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_compound_elasticities_df() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_compound_elasticities_df(
#         model=model,
#         compounds=compounds,
#         y=y0,
#         t=0,
#         normalized=False,
#     )
#     self.assertEqual(list(df.index), ["X", "Y"])
#     self.assertEqual(list(df.columns), ["v_in", "v1", "v_out"])
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#     )


# def test_get_compound_elasticities_df_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_compound_elasticities_df(
#         model=model,
#         compounds=compounds,
#         y=y0,
#         t=0,
#         normalized=True,
#     )
#     self.assertEqual(list(df.index), ["X", "Y"])
#     self.assertEqual(list(df.columns), ["v_in", "v1", "v_out"])
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_parameter_elasticity() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticity(
#             model=model,
#             parameter="k_in",
#             y=y0,
#             t=0,
#             normalized=False,
#         ),
#         [1.0, 0.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticity(
#             model=model,
#             parameter="k_in",
#             y=y0,
#             t=0,
#             normalized=True,
#         ),
#         [1.0, 0.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticity(
#             model=model,
#             parameter="k_fwd",
#             y=y0,
#             t=0,
#             normalized=False,
#         ),
#         [0.0, 1.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticity(
#             model=model,
#             parameter="k_fwd",
#             y=y0,
#             t=0,
#             normalized=True,
#         ),
#         [0.0, 1.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticity(
#             model=model,
#             parameter="k_out",
#             y=y0,
#             t=0,
#             normalized=False,
#         ),
#         [0.0, 0.0, 1.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticity(
#             model=model,
#             parameter="k_out",
#             y=y0,
#             t=0,
#             normalized=True,
#         ),
#         [0.0, 0.0, 1.0],
#         decimal=4,
#     )


# def test_get_parameter_elasticities_array() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticities_array(
#             model=model,
#             parameters=parameters,
#             y=y0,
#             t=0,
#             normalized=False,
#         ),
#         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_parameter_elasticities_array_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_parameter_elasticities_array(
#             model=model,
#             parameters=parameters,
#             y=y0,
#             t=0,
#             normalized=True,
#         ),
#         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_parameter_elasticities_df() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_parameter_elasticities_df(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         t=0,
#         normalized=False,
#     )
#     self.assertEqual(list(df.index), ["k_in", "k_fwd", "k_out"])
#     self.assertEqual(list(df.columns), ["v_in", "v1", "v_out"])
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_parameter_elasticities_df_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_parameter_elasticities_df(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         t=0,
#         normalized=True,
#     )
#     self.assertEqual(list(df.index), ["k_in", "k_fwd", "k_out"])
#     self.assertEqual(list(df.columns), ["v_in", "v1", "v_out"])
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         decimal=4,
#     )


# def test_get_concentration_response_coefficient() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_concentration_response_coefficient(
#             model=model,
#             parameter="k_in",
#             y=y0,
#             normalized=False,
#         ),
#         [1, 1],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_concentration_response_coefficient(
#             model=model,
#             parameter="k_in",
#             y=y0,
#             normalized=True,
#         ),
#         [1, 1],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_concentration_response_coefficient(
#             model=model,
#             parameter="k_fwd",
#             y=y0,
#             normalized=False,
#         ),
#         [-1, 0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_concentration_response_coefficient(
#             model=model,
#             parameter="k_fwd",
#             y=y0,
#             normalized=True,
#         ),
#         [-1, 0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_concentration_response_coefficient(
#             model=model,
#             parameter="k_out",
#             y=y0,
#             normalized=False,
#         ),
#         [0, -1],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_concentration_response_coefficient(
#             model=model,
#             parameter="k_out",
#             y=y0,
#             normalized=True,
#         ),
#         [0, -1],
#         decimal=4,
#     )


# def test_get_concentration_response_coefficients_array() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     arr = mca.get_concentration_response_coefficients_array(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     np.testing.assert_array_almost_equal(
#         arr,
#         [
#             [1, 1],
#             [-1, 0],
#             [0, -1],
#         ],
#         decimal=4,
#     )


# def test_get_concentration_response_coefficients_array_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     arr = mca.get_concentration_response_coefficients_array(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=True,
#     )
#     np.testing.assert_array_almost_equal(
#         arr,
#         [
#             [1, 1],
#             [-1, 0],
#             [0, -1],
#         ],
#         decimal=4,
#     )


# def test_get_concentration_response_coefficients_df() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_concentration_response_coefficients_df(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [
#             [1, 1],
#             [-1, 0],
#             [0, -1],
#         ],
#         decimal=4,
#     )
#     self.assertEqual(list(df.index), ["k_in", "k_fwd", "k_out"])
#     self.assertEqual(list(df.columns), ["X", "Y"])


# def test_get_concentration_response_coefficients_df_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_concentration_response_coefficients_df(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=True,
#     )
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [
#             [1, 1],
#             [-1, 0],
#             [0, -1],
#         ],
#         decimal=4,
#     )
#     self.assertEqual(list(df.index), ["k_in", "k_fwd", "k_out"])
#     self.assertEqual(list(df.columns), ["X", "Y"])


# def test_get_flux_response_coefficient() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     np.testing.assert_array_almost_equal(
#         mca.get_flux_response_coefficient(
#             model=model,
#             parameter="k_in",
#             y=y0,
#             normalized=False,
#         ),
#         [1, 1, 1],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_flux_response_coefficient(
#             model=model,
#             parameter="k_in",
#             y=y0,
#             normalized=True,
#         ),
#         [1, 1, 1],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_flux_response_coefficient(
#             model=model,
#             parameter="k_fwd",
#             y=y0,
#             normalized=False,
#         ),
#         [0.0, 0.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_flux_response_coefficient(
#             model=model,
#             parameter="k_fwd",
#             y=y0,
#             normalized=True,
#         ),
#         [0.0, 0.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_flux_response_coefficient(
#             model=model,
#             parameter="k_out",
#             y=y0,
#             normalized=False,
#         ),
#         [0.0, 0.0, 0.0],
#         decimal=4,
#     )
#     np.testing.assert_array_almost_equal(
#         mca.get_flux_response_coefficient(
#             model=model,
#             parameter="k_out",
#             y=y0,
#             normalized=True,
#         ),
#         [0.0, 0.0, 0.0],
#         decimal=4,
#     )


# def test_get_flux_response_coefficients_array() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     arr = mca.get_flux_response_coefficients_array(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     np.testing.assert_array_almost_equal(
#         arr,
#         [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#         decimal=4,
#     )


# def test_get_flux_response_coefficients_array_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     arr = mca.get_flux_response_coefficients_array(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     np.testing.assert_array_almost_equal(
#         arr,
#         [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#         decimal=4,
#     )


# def test_get_flux_response_coefficients_df() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_flux_response_coefficients_df(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     self.assertEqual(list(df.index), ["k_in", "k_fwd", "k_out"])
#     self.assertEqual(list(df.columns), ["v_in", "v1", "v_out"])
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#         decimal=4,
#     )


# def test_get_flux_response_coefficients_df_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     df = mca.get_flux_response_coefficients_df(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=True,
#     )
#     self.assertEqual(list(df.index), ["k_in", "k_fwd", "k_out"])
#     self.assertEqual(list(df.columns), ["v_in", "v1", "v_out"])
#     np.testing.assert_array_almost_equal(
#         df.values,
#         [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#         decimal=4,
#     )


# def test_plot_concentration_response_coefficients() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     mca.plot_concentration_response_coefficients(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     plt.close()


# def test_plot_concentration_response_coefficients_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     mca.plot_concentration_response_coefficients(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=True,
#     )
#     plt.close()


# def test_plot_flux_response_coefficients() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     mca.plot_flux_response_coefficients(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=False,
#     )
#     plt.close()


# def test_plot_flux_response_coefficients_normalized() -> None:
#     model, y0, parameters, compounds = self.create_model()
#     mca.plot_flux_response_coefficients(
#         model=model,
#         parameters=parameters,
#         y=y0,
#         normalized=True,
#     )
#     plt.close()


# def test_parallel_get_response_coefficients_array() -> None:
#     model, y0, parameters, compounds = self.create_model()

#     crc1, frc1 = mca.get_response_coefficients_array(
#         model=model,
#         parameters=[k for k in parameters[:3]],
#         y=y0,
#         multiprocessing=False,
#     )

#     crc2, frc2 = mca.get_response_coefficients_array(
#         model=model,
#         parameters=[k for k in parameters[:3]],
#         y=y0,
#         multiprocessing=True,
#     )

#     np.testing.assert_array_almost_equal(crc1, crc2, decimal=4)
#     np.testing.assert_array_almost_equal(frc1, frc2, decimal=4)
