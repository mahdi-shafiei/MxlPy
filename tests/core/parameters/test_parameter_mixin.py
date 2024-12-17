# type: ignore

from __future__ import annotations

import tempfile
import unittest

from modelbase.ode import Model


class ModelBasicTests(unittest.TestCase):
    def test_init_empty(self):
        model = Model()
        self.assertEqual(model.parameters, {})
        self.assertEqual(model.parameters, model.initialization_parameters)
        self.assertIsNot(model.parameters, model.initialization_parameters)

    def test_init_parameters(self):
        parameters = {"k1": 1}
        model = Model(parameters=parameters)
        self.assertEqual(model.parameters, parameters)
        self.assertEqual(model.parameters, model.initialization_parameters)
        self.assertIsNot(model.parameters, model.initialization_parameters)

    def test_enter(self):
        parameters = {"k1": 1}
        model = Model(parameters=parameters)
        with model as m_dup:
            m_dup.update_parameter("k1", 2)
            self.assertEqual(m_dup.parameters["k1"], 2)
            self.assertEqual(model.parameters["k1"], 1)

    def test_exit(self):
        parameters = {"k1": 1}
        model = Model(parameters=parameters)
        with model:
            model.update_parameter("k1", 2)
            self.assertEqual(model.parameters["k1"], 2)
        self.assertEqual(model.parameters["k1"], 1)

    def test_copy(self):
        parameters = {"k1": 1}
        m1 = Model(parameters=parameters)
        m2 = m1.copy()
        self.assertIsNot(m1, m2)
        self.assertIsNot(m1.parameters, m2.parameters)
        self.assertEqual(m1.parameters, m2.parameters)


class ModelTests(unittest.TestCase):
    def test_add_parameter(self):
        model = Model()
        model.add_parameter(parameter_name="k1", parameter_value=1)
        self.assertEqual(model.parameters["k1"], 1)

    def test_add_parameter_fail_on_existing(self):
        model = Model()
        model.add_parameter(parameter_name="k1", parameter_value=1)
        with self.assertWarns(UserWarning):
            model.add_parameter(parameter_name="k1", parameter_value=1)

    def test_add_parameters(self):
        model = Model()
        model.add_parameters(parameters={"k1": 1, "k2": 2})
        self.assertEqual(model.parameters["k1"], 1)
        self.assertEqual(model.parameters["k2"], 2)

    def test_update_parameter(self):
        parameters = {"k1": 1}
        model = Model(parameters=parameters)
        model.update_parameter(parameter_name="k1", parameter_value=2)
        self.assertTrue(model.parameters["k1"], 2)

    def test_update_parameter_fail_on_new(self):
        model = Model()
        with self.assertWarns(UserWarning):
            model.update_parameter(parameter_name="k1", parameter_value=2)

    def test_update_parameters(self):
        model = Model()
        model.add_parameters(parameters={"k1": 1, "k2": 2})
        model.update_parameters(parameters={"k1": 2, "k2": 3})
        self.assertEqual(model.parameters["k1"], 2)
        self.assertEqual(model.parameters["k2"], 3)

    def test_add_and_update_parameter_new(self):
        model = Model()
        model.add_and_update_parameter(parameter_name="k1", parameter_value=1)
        self.assertEqual(model.parameters["k1"], 1)

    def test_add_and_update_parameter_existing(self):
        model = Model()
        model.add_parameter(parameter_name="k1", parameter_value=1)
        model.add_and_update_parameter(parameter_name="k1", parameter_value=2)
        self.assertEqual(model.parameters["k1"], 2)

    def test_add_and_update_parameters(self):
        model = Model()
        model.add_parameter(parameter_name="k1", parameter_value=1)
        model.add_and_update_parameters(parameters={"k1": 2, "k2": 3})
        self.assertEqual(model.parameters["k1"], 2)
        self.assertEqual(model.parameters["k2"], 3)

    def test_remove_parameter(self):
        parameters = {"k1": 1, "k2": 2}
        model = Model(parameters=parameters)
        model.remove_parameter(parameter_name="k1")
        self.assertEqual(model.parameters, {"k2": 2})

    def test_remove_parameters(self):
        parameters = {"k1": 1, "k2": 2}
        model = Model(parameters=parameters)
        model.remove_parameters(parameter_names=["k1", "k2"])
        self.assertEqual(model.parameters, {})

    def test_store_and_load_parameters_new_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model()
            m1.store_parameters_to_file(filename=f"{tmpdir}/test", filetype="json")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.json", filetype="json")
            self.assertEqual(m1.parameters, m2.parameters)
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model()
            m1.store_parameters_to_file(filename=f"{tmpdir}/test.json", filetype="json")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.json", filetype="json")
            self.assertEqual(m1.parameters, m2.parameters)

    def test_store_and_load_parameters_new_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model()
            m1.store_parameters_to_file(filename=f"{tmpdir}/test", filetype="pickle")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.p", filetype="pickle")
            self.assertEqual(m1.parameters, m2.parameters)
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model()
            m1.store_parameters_to_file(filename=f"{tmpdir}/test.p", filetype="pickle")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.p", filetype="pickle")
            self.assertEqual(m1.parameters, m2.parameters)

    def test_store_and_load_parameters_existing_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model(parameters=parameters)
            m1.store_parameters_to_file(filename=f"{tmpdir}/test", filetype="json")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.json", filetype="json")
            self.assertEqual(m1.parameters, m2.parameters)
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model(parameters=parameters)
            m1.store_parameters_to_file(filename=f"{tmpdir}/test.json", filetype="json")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.json", filetype="json")
            self.assertEqual(m1.parameters, m2.parameters)

    def test_store_and_load_parameters_existing_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model(parameters=parameters)
            m1.store_parameters_to_file(filename=f"{tmpdir}/test", filetype="pickle")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.p", filetype="pickle")
            self.assertEqual(m1.parameters, m2.parameters)
        with tempfile.TemporaryDirectory() as tmpdir:
            parameters = {"k1": 1, "k2": 2}
            m1 = Model(parameters=parameters)
            m2 = Model(parameters=parameters)
            m1.store_parameters_to_file(filename=f"{tmpdir}/test.p", filetype="pickle")
            m2.load_parameters_from_file(filename=f"{tmpdir}/test.p", filetype="pickle")
            self.assertEqual(m1.parameters, m2.parameters)

    def test_store_and_load_parameters_fail_on_other(self):
        model = Model()
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdir:
                model.store_parameters_to_file(filename=f"{tmpdir}/test", filetype="xml")

        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdir:
                model.load_parameters_from_file(filename=f"{tmpdir}/test", filetype="xml")

    def test_restore_initialization_parameters(self):
        parameters = {"k1": 1}
        model = Model(parameters=parameters)
        model.update_parameter("k1", 5)
        model.add_parameter("k2", 2)
        model.restore_initialization_parameters()
        self.assertEqual(model.parameters["k1"], 1)
        with self.assertRaises(KeyError):
            model.parameters["k2"]

    def test_get_parameter(self):
        parameters = {"k1": 1}
        model = Model(parameters=parameters)
        self.assertEqual(model.get_parameter(parameter_name="k1"), 1)

    def test_get_parameters(self):
        parameters = {"k1": 1, "k2": 2}
        model = Model(parameters=parameters)
        self.assertEqual(model.get_parameters(), {"k1": 1, "k2": 2})

    ############################################################################
    # Updating meta info
    ############################################################################

    # def test_update_parameter_meta_info(self):
    #     model = Model()
    #     model.add_parameter(parameter_name="x", parameter_value=0)
    #     model.update_parameter_meta_info(parameter="x", meta_info={"unit": "X"})
    #     self.assertEqual(model.meta_info["parameters"]["x"].unit, "X")

    # def test_update_parameter_meta_info_replacing(self):
    #     model = Model()
    #     model.add_parameter(parameter_name="x", parameter_value=0, **{"unit": "X1"})
    #     model.update_parameter_meta_info(parameter="x", meta_info={"unit": "X2"})
    #     self.assertEqual(model.meta_info["parameters"]["x"].unit, "X2")


class DerivedParameterTests(unittest.TestCase):
    def test_add_derived_parameter(self):
        model = Model()
        model.add_parameters({"k_fwd": 1, "k_eq": 10})
        model.add_derived_parameter(
            parameter_name="k_bwd",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        self.assertEqual(model.parameters, {"k_fwd": 1, "k_eq": 10, "k_bwd": 0.1})
        self.assertEqual(
            model.derived_parameters["k_bwd"]["parameters"], ["k_fwd", "k_eq"]
        )
        self.assertEqual(model._derived_from_parameters, {"k_fwd", "k_eq"})

    def test_remove_derived_parameter(self):
        model = Model()
        model.add_parameters({"k_fwd": 1, "k_eq": 10})
        model.add_derived_parameter(
            parameter_name="k_bwd",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        model.remove_derived_parameter("k_bwd")
        self.assertEqual(model.parameters, {"k_fwd": 1, "k_eq": 10})
        self.assertEqual(model.derived_parameters, {})
        self.assertEqual(model._derived_from_parameters, set())

    def test_remove_derived_parameter_multiple_dependencies(self):
        model = Model()
        model.add_parameters({"k_fwd": 1, "k_eq": 10})
        model.add_derived_parameter(
            parameter_name="k_bwd",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        model.add_derived_parameter(
            parameter_name="k_bwd2",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        model.remove_derived_parameter("k_bwd")
        self.assertEqual(model.parameters, {"k_fwd": 1, "k_eq": 10, "k_bwd2": 0.1})
        self.assertEqual(model._derived_from_parameters, {"k_fwd", "k_eq"})

    def test_update_derived_parameters_on_update(self):
        model = Model()
        model.add_parameters({"k_fwd": 1, "k_eq": 10})
        model.add_derived_parameter(
            parameter_name="k_bwd",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        model.update_parameter("k_fwd", 2)
        self.assertEqual(model.parameters["k_bwd"], 0.2)

    def test_update_derived_parameters_on_add(self):
        model = Model()
        model.add_parameters({"k_fwd": 1, "k_eq": 10})
        model.add_derived_parameter(
            parameter_name="k_bwd",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        with self.assertWarns(UserWarning):
            model.add_parameter("k_fwd", 2)
        self.assertEqual(model.parameters["k_bwd"], 0.2)

    def test_update_derived_parameters_on_add_and_update(self):
        model = Model()
        model.add_parameters({"k_fwd": 1, "k_eq": 10})
        model.add_derived_parameter(
            parameter_name="k_bwd",
            function=lambda kf, keq: kf / keq,
            parameters=["k_fwd", "k_eq"],
        )
        model.add_and_update_parameter("k_fwd", 2)
        self.assertEqual(model.parameters["k_bwd"], 0.2)


class SBMLTests(unittest.TestCase):
    def test_create_sbml_parameters_without_meta_info(self):
        model = Model()
        model.add_parameter("k_in", 1)
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_parameters(sbml_model=sbml_model)
        parameter = sbml_model.getListOfParameters()[0]
        self.assertEqual(parameter.getId(), "k_in")
        self.assertEqual(parameter.getValue(), 1.0)
        self.assertEqual(parameter.getConstant(), True)
        self.assertEqual(parameter.getUnits(), "")

    def test_create_sbml_parameters_with_meta_info(self):
        model = Model()
        model.add_parameter("k_in", 1, **{"unit": "mM"})
        doc = model._create_sbml_document()
        sbml_model = model._create_sbml_model(doc=doc)
        model._create_sbml_parameters(sbml_model=sbml_model)
        parameter = sbml_model.getListOfParameters()[0]
        self.assertEqual(parameter.getId(), "k_in")
        self.assertEqual(parameter.getValue(), 1.0)
        self.assertEqual(parameter.getConstant(), True)
        self.assertEqual(parameter.getUnits(), "mM")
