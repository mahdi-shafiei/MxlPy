# type: ignore

from __future__ import annotations

import unittest

from modelbase.ode import LabelModel
from modelbase.ode import ratefunctions as rf


class LabelModelTests(unittest.TestCase):
    def test_generate_binary_labels_no_label(self):
        self.assertEqual(
            LabelModel._generate_binary_labels(base_name="name", num_labels=0),
            ["name"],
        )

    def test_generate_binary_labels_one_label(self):
        self.assertEqual(
            LabelModel._generate_binary_labels(base_name="name", num_labels=1),
            ["name__0", "name__1"],
        )

    def test_generate_binary_labels_two_labels(self):
        self.assertEqual(
            LabelModel._generate_binary_labels(base_name="name", num_labels=2),
            ["name__00", "name__01", "name__10", "name__11"],
        )

    def test_add_compound(self):
        label_model = LabelModel()
        label_model.add_compound(compound="name", is_isotopomer=False)
        self.assertEqual(label_model.label_compounds, {})
        self.assertEqual(label_model.nonlabel_compounds, ["name"])
        self.assertEqual(label_model.compounds, ["name"])

    def test_add_compound_isotopomer(self):
        label_model = LabelModel()
        label_model.add_compound(compound="name", is_isotopomer=True)
        self.assertEqual(label_model.label_compounds, {})
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(label_model.compounds, ["name"])

    def test_add_label_compound_meta_info(self):
        label_model = LabelModel()
        label_model.add_label_compound(
            compound="name",
            num_labels=1,
            **{
                "common_name": "cpd",
                "compartment": "c",
                "formula": "C1",
                "charge": 2,
                "gibbs0": 2,
                "smiles": "abc",
                "database_links": {"metacyc": "cpd-1"},
                "notes": {"test": "done"},
            },
        )
        self.assertEqual(label_model.label_compounds["name"]["num_labels"], 1)
        self.assertEqual(
            label_model.label_compounds["name"]["isotopomers"],
            ["name__0", "name__1"],
        )
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(label_model.compounds, ["name__0", "name__1"])

        self.assertEqual(list(label_model.meta_info["compounds"].keys()), ["name"])
        cpd = label_model.meta_info["compounds"]["name"]
        self.assertEqual(cpd.common_name, "cpd")
        self.assertEqual(cpd.compartment, "c")
        self.assertEqual(cpd.formula, "C1")
        self.assertEqual(cpd.charge, 2)
        self.assertEqual(cpd.gibbs0, 2)
        self.assertEqual(cpd.smiles, "abc")
        self.assertEqual(cpd.database_links, {"metacyc": "cpd-1"})
        self.assertEqual(cpd.notes, {"test": "done"})

    def test_add_label_compound_no_label(self):
        label_model = LabelModel()
        label_model.add_label_compound(compound="name", num_labels=0)
        self.assertEqual(label_model.label_compounds, {})
        self.assertEqual(label_model.nonlabel_compounds, ["name"])
        self.assertEqual(label_model.compounds, ["name"])

    def test_add_label_compound_one_label(self):
        label_model = LabelModel()
        label_model.add_label_compound(compound="name", num_labels=1)
        self.assertEqual(label_model.label_compounds["name"]["num_labels"], 1)
        self.assertEqual(
            label_model.label_compounds["name"]["isotopomers"],
            ["name__0", "name__1"],
        )
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(label_model.compounds, ["name__0", "name__1"])

    def test_add_label_compound_two_labels(self):
        label_model = LabelModel()
        label_model.add_label_compound(compound="name", num_labels=2)
        self.assertEqual(label_model.label_compounds["name"]["num_labels"], 2)
        self.assertEqual(
            label_model.label_compounds["name"]["isotopomers"],
            ["name__00", "name__01", "name__10", "name__11"],
        )
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(
            label_model.compounds,
            ["name__00", "name__01", "name__10", "name__11"],
        )

    def test_add_label_compound_overwrite(self):
        label_model = LabelModel()
        label_model.add_label_compound(compound="name", num_labels=2)
        with self.assertWarns(UserWarning):
            label_model.add_label_compound(compound="name", num_labels=1)
        self.assertEqual(label_model.label_compounds["name"]["num_labels"], 1)
        self.assertEqual(
            label_model.label_compounds["name"]["isotopomers"],
            ["name__0", "name__1"],
        )
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(label_model.compounds, ["name__0", "name__1"])

    def test_add_label_compounds(self):
        label_model = LabelModel()
        label_model.add_label_compounds(compounds={"a": 1, "b": 2})
        self.assertEqual(label_model.label_compounds["a"]["num_labels"], 1)
        self.assertEqual(label_model.label_compounds["b"]["num_labels"], 2)
        self.assertEqual(
            label_model.label_compounds["a"]["isotopomers"],
            ["a__0", "a__1"],
        )
        self.assertEqual(
            label_model.label_compounds["b"]["isotopomers"],
            ["b__00", "b__01", "b__10", "b__11"],
        )
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(
            label_model.compounds,
            ["a__0", "a__1", "b__00", "b__01", "b__10", "b__11"],
        )

    def test_remove_compound(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd", num_labels=2)
        label_model.remove_compound(compound="cpd")
        self.assertEqual(
            label_model.compounds,
            [
                "label-cpd__00",
                "label-cpd__01",
                "label-cpd__10",
                "label-cpd__11",
            ],
        )
        self.assertEqual(label_model.nonlabel_compounds, [])
        self.assertEqual(label_model.label_compounds["label-cpd"]["num_labels"], 2)
        self.assertEqual(
            label_model.label_compounds["label-cpd"]["isotopomers"],
            [
                "label-cpd__00",
                "label-cpd__01",
                "label-cpd__10",
                "label-cpd__11",
            ],
        )

    def test_remove_label_compound(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd", num_labels=2)
        label_model.remove_label_compound(compound="label-cpd")
        self.assertEqual(
            label_model.compounds,
            ["cpd"],
        )
        self.assertEqual(label_model.nonlabel_compounds, ["cpd"])
        self.assertEqual(label_model.label_compounds, {})

    def test_remove_label_compounds(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd-1", num_labels=2)
        label_model.add_label_compound(compound="label-cpd-2", num_labels=3)
        label_model.remove_label_compounds(compounds=["label-cpd-1", "label-cpd-2"])
        self.assertEqual(
            label_model.compounds,
            ["cpd"],
        )
        self.assertEqual(label_model.nonlabel_compounds, ["cpd"])
        self.assertEqual(label_model.label_compounds, {})

    def test_get_base_compounds(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd-1", num_labels=2)
        label_model.add_label_compound(compound="label-cpd-2", num_labels=3)
        self.assertEqual(
            label_model.get_base_compounds(),
            ["cpd", "label-cpd-1", "label-cpd-2"],
        )

    def test_get_compound_number_of_label_positions(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd-1", num_labels=2)
        label_model.add_label_compound(compound="label-cpd-2", num_labels=3)
        self.assertEqual(
            label_model.get_compound_number_of_label_positions(compound="label-cpd-1"),
            2,
        )
        self.assertEqual(
            label_model.get_compound_number_of_label_positions(compound="label-cpd-2"),
            3,
        )

    def test_get_compound_isotopomers(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd-1", num_labels=2)
        label_model.add_label_compound(compound="label-cpd-2", num_labels=3)
        self.assertEqual(
            label_model.get_compound_isotopomers(compound="label-cpd-1"),
            [
                "label-cpd-1__00",
                "label-cpd-1__01",
                "label-cpd-1__10",
                "label-cpd-1__11",
            ],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers(compound="label-cpd-2"),
            [
                "label-cpd-2__000",
                "label-cpd-2__001",
                "label-cpd-2__010",
                "label-cpd-2__011",
                "label-cpd-2__100",
                "label-cpd-2__101",
                "label-cpd-2__110",
                "label-cpd-2__111",
            ],
        )

    def test_get_compound_isotopomers_with_n_labels(self):
        label_model = LabelModel()
        label_model.add_compound(compound="cpd")
        label_model.add_label_compound(compound="label-cpd-1", num_labels=2)
        label_model.add_label_compound(compound="label-cpd-2", num_labels=3)
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-1", n_labels=0
            ),
            ["label-cpd-1__00"],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-1", n_labels=1
            ),
            ["label-cpd-1__10", "label-cpd-1__01"],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-1", n_labels=2
            ),
            ["label-cpd-1__11"],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-2", n_labels=0
            ),
            ["label-cpd-2__000"],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-2", n_labels=1
            ),
            [
                "label-cpd-2__100",
                "label-cpd-2__010",
                "label-cpd-2__001",
            ],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-2", n_labels=2
            ),
            [
                "label-cpd-2__110",
                "label-cpd-2__101",
                "label-cpd-2__011",
            ],
        )
        self.assertEqual(
            label_model.get_compound_isotopomers_with_n_labels(
                compound="label-cpd-2", n_labels=3
            ),
            ["label-cpd-2__111"],
        )

    def test_add_algebraic_module_total(self):
        label_model = LabelModel()
        label_model.add_compound("x")
        label_model.add_label_compound("y", 2)
        label_model.add_algebraic_module(
            module_name="mod1",
            function=lambda *args: tuple(args),
            compounds=["x", "y"],
            derived_compounds=None,
            modifiers=["x", "y"],
            parameters=None,
        )
        self.assertEqual(
            label_model.algebraic_modules["mod1"]["compounds"], ["x", "y__total"]
        )
        self.assertEqual(
            label_model.algebraic_modules["mod1"]["modifiers"], ["x", "y__total"]
        )

    def test_add_algebraic_module_specific_isotopomer(self):
        label_model = LabelModel()
        label_model.add_compound("x")
        label_model.add_label_compound("y", 2)
        label_model.add_algebraic_module(
            module_name="mod1",
            function=lambda *args: tuple(args),
            compounds=["x", "y__0"],
            derived_compounds=None,
            modifiers=["x", "y__0"],
            parameters=None,
        )
        self.assertEqual(
            label_model.algebraic_modules["mod1"]["compounds"], ["x", "y__0"]
        )
        self.assertEqual(
            label_model.algebraic_modules["mod1"]["modifiers"], ["x", "y__0"]
        )

    def test_split_label_string_two_one_label_compounds(self):
        res = LabelModel._split_label_string(label="01", labels_per_compound=[1, 1])
        self.assertEqual(res, ["0", "1"])

    def test_split_label_string_one_two_label_compound(self):
        res = LabelModel._split_label_string(label="01", labels_per_compound=[2])
        self.assertEqual(res, ["01"])

    def test_split_label_string_2_2(self):
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[4]),
            ["0011"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[3, 1]),
            ["001", "1"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[2, 2]),
            ["00", "11"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[1, 3]),
            ["0", "011"],
        )

    def test_split_label_string_2_2_with_nonlabel_compounds(self):
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[0, 4]),
            ["", "0011"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[4, 0]),
            ["0011", ""],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[0, 3, 1]),
            ["", "001", "1"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[3, 0, 1]),
            ["001", "", "1"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[3, 1, 0]),
            ["001", "1", ""],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[0, 2, 2]),
            ["", "00", "11"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[2, 0, 2]),
            ["00", "", "11"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[2, 2, 0]),
            [
                "00",
                "11",
                "",
            ],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[0, 1, 3]),
            ["", "0", "011"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[1, 0, 3]),
            ["0", "", "011"],
        )
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[1, 3, 0]),
            ["0", "011", ""],
        )

    def test_split_label_string_2_2_with_multiple_nonlabel_compounds(
        self,
    ):
        self.assertEqual(
            LabelModel._split_label_string(label="0011", labels_per_compound=[0, 4, 0]),
            ["", "0011", ""],
        )

    def test_map_substrates_to_products_4(self):
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0000", labelmap=[0, 1, 2, 3]
            ),
            "0000",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0001", labelmap=[0, 1, 2, 3]
            ),
            "0001",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0010", labelmap=[0, 1, 2, 3]
            ),
            "0010",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0011", labelmap=[0, 1, 2, 3]
            ),
            "0011",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0100", labelmap=[0, 1, 2, 3]
            ),
            "0100",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0101", labelmap=[0, 1, 2, 3]
            ),
            "0101",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0110", labelmap=[0, 1, 2, 3]
            ),
            "0110",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0111", labelmap=[0, 1, 2, 3]
            ),
            "0111",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1000", labelmap=[0, 1, 2, 3]
            ),
            "1000",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1001", labelmap=[0, 1, 2, 3]
            ),
            "1001",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1010", labelmap=[0, 1, 2, 3]
            ),
            "1010",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1011", labelmap=[0, 1, 2, 3]
            ),
            "1011",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1100", labelmap=[0, 1, 2, 3]
            ),
            "1100",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1101", labelmap=[0, 1, 2, 3]
            ),
            "1101",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1110", labelmap=[0, 1, 2, 3]
            ),
            "1110",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1111", labelmap=[0, 1, 2, 3]
            ),
            "1111",
        )

    def test_map_substrates_to_products_4_inverted(self):
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0000", labelmap=[3, 2, 1, 0]
            ),
            "0000",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0001", labelmap=[3, 2, 1, 0]
            ),
            "1000",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0010", labelmap=[3, 2, 1, 0]
            ),
            "0100",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0011", labelmap=[3, 2, 1, 0]
            ),
            "1100",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0100", labelmap=[3, 2, 1, 0]
            ),
            "0010",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0101", labelmap=[3, 2, 1, 0]
            ),
            "1010",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0110", labelmap=[3, 2, 1, 0]
            ),
            "0110",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="0111", labelmap=[3, 2, 1, 0]
            ),
            "1110",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1000", labelmap=[3, 2, 1, 0]
            ),
            "0001",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1001", labelmap=[3, 2, 1, 0]
            ),
            "1001",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1010", labelmap=[3, 2, 1, 0]
            ),
            "0101",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1011", labelmap=[3, 2, 1, 0]
            ),
            "1101",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1100", labelmap=[3, 2, 1, 0]
            ),
            "0011",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1101", labelmap=[3, 2, 1, 0]
            ),
            "1011",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1110", labelmap=[3, 2, 1, 0]
            ),
            "0111",
        )
        self.assertEqual(
            LabelModel._map_substrates_to_products(
                rate_suffix="1111", labelmap=[3, 2, 1, 0]
            ),
            "1111",
        )

    def test_unpack_stoichiometries(self):
        self.assertEqual(
            LabelModel._unpack_stoichiometries(stoichiometries={"x": -1, "y": 1}),
            (["x"], ["y"]),
        )
        self.assertEqual(
            LabelModel._unpack_stoichiometries(stoichiometries={"x": -2, "y": 1}),
            (["x", "x"], ["y"]),
        )
        self.assertEqual(
            LabelModel._unpack_stoichiometries(stoichiometries={"x": -1, "y": 2}),
            (["x"], ["y", "y"]),
        )
        self.assertEqual(
            LabelModel._unpack_stoichiometries(stoichiometries={"y": 1, "x": -1}),
            (["x"], ["y"]),
        )

    def test_get_labels_per_compound(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 3})
        self.assertEqual(
            label_model._get_labels_per_compound(compounds=["x", "y"]),
            [2, 3],
        )

    def test_get_labels_per_compound_nonlabeled(self):
        label_model = LabelModel()
        label_model.add_compound("non-label")
        label_model.add_label_compounds({"x": 2, "y": 3})
        self.assertEqual(
            label_model._get_labels_per_compound(compounds=["non-label", "x", "y"]),
            [0, 2, 3],
        )
        self.assertEqual(
            label_model._get_labels_per_compound(compounds=["x", "non-label", "y"]),
            [2, 0, 3],
        )
        self.assertEqual(
            label_model._get_labels_per_compound(compounds=["x", "y", "non-label"]),
            [2, 3, 0],
        )

    def test_repack_stoichiometries_empty(self):
        self.assertEqual(
            LabelModel._repack_stoichiometries(new_substrates=[], new_products=[]),
            {},
        )

    def test_repack_stoichiometries_1_1(self):
        self.assertEqual(
            LabelModel._repack_stoichiometries(new_substrates=["x"], new_products=["y"]),
            {"x": -1, "y": 1},
        )

    def test_repack_stoichiometries_2_2(self):
        self.assertEqual(
            LabelModel._repack_stoichiometries(
                new_substrates=["x", "x"], new_products=["y", "y"]
            ),
            {"x": -2, "y": 2},
        )

    def test_assign_compound_labels_no_label(self):
        self.assertEqual(
            LabelModel._assign_compound_labels(base_compounds=["x"], label_suffixes=[""]),
            ["x"],
        )

    def test_assign_compound_labels_one_label(self):
        self.assertEqual(
            LabelModel._assign_compound_labels(
                base_compounds=["x"], label_suffixes=["0"]
            ),
            ["x__0"],
        )

    def test_assign_compound_labels(self):
        self.assertEqual(
            LabelModel._assign_compound_labels(
                base_compounds=["x", "y"], label_suffixes=["0", "1"]
            ),
            ["x__0", "y__1"],
        )

    def test_add_labelmap_reaction_meta_info(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0],
            **{
                "common_name": "reaction",
                "gibbs0": 1,
                "ec": "123",
                "database_links": {"metacyc": 1},
                "notes": {"test": 1},
                "sbml_function": "a * b",
            },
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0": {"x__0": -1, "y__0": 1},
                "v1__1": {"x__1": -1, "y__1": 1},
            },
        )

        self.assertEqual(list(label_model.meta_info["rates"].keys()), ["v1"])
        rate = label_model.meta_info["rates"]["v1"]
        self.assertEqual(rate.common_name, "reaction")
        self.assertEqual(rate.gibbs0, 1)
        self.assertEqual(rate.ec, "123")
        self.assertEqual(rate.database_links, {"metacyc": 1})
        self.assertEqual(rate.notes, {"test": 1})
        self.assertEqual(rate.sbml_function, "a * b")

    def test_add_labelmap_reaction_no_meta_info(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0": {"x__0": -1, "y__0": 1},
                "v1__1": {"x__1": -1, "y__1": 1},
            },
        )

        self.assertEqual(list(label_model.meta_info["rates"].keys()), ["v1"])
        rate = label_model.meta_info["rates"]["v1"]
        self.assertEqual(rate.common_name, None)
        self.assertEqual(rate.gibbs0, None)
        self.assertEqual(rate.ec, None)
        self.assertEqual(rate.database_links, {})
        self.assertEqual(rate.notes, {})
        self.assertEqual(rate.sbml_function, None)

    def test_add_labelmap_reaction_1_1(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0": {"x__0": -1, "y__0": 1},
                "v1__1": {"x__1": -1, "y__1": 1},
            },
        )

    def test_add_labelmap_reaction_1_1_total_modifier(self):
        label_model = LabelModel()
        label_model.add_compounds(["ATP", "ADP"])
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"ATP": -1, "x": -1, "ADP": -1, "y": 1},
            labelmap=[0],
            modifiers=["x"],
        )
        self.assertEqual(label_model.rates["v1__0"]["modifiers"], ["x__total"])
        self.assertEqual(label_model.rates["v1__1"]["modifiers"], ["x__total"])

    def test_add_labelmap_reaction_1_1_specific_modifier(self):
        label_model = LabelModel()
        label_model.add_compounds(["ATP", "ADP"])
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"ATP": -1, "x": -1, "ADP": -1, "y": 1},
            labelmap=[0],
            modifiers=["x__00"],
        )
        self.assertEqual(label_model.rates["v1__0"]["modifiers"], ["x__00"])
        self.assertEqual(label_model.rates["v1__1"]["modifiers"], ["x__00"])

    def test_add_labelmap_reaction_1_1_cofactor_first_place(self):
        label_model = LabelModel()
        label_model.add_compounds(["ATP", "ADP"])
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"ATP": -1, "x": -1, "ADP": -1, "y": 1},
            labelmap=[0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0": {
                    "ATP": -1,
                    "x__0": -1,
                    "ADP": -1,
                    "y__0": 1,
                },
                "v1__1": {
                    "ATP": -1,
                    "x__1": -1,
                    "ADP": -1,
                    "y__1": 1,
                },
            },
        )

    def test_add_labelmap_reaction_1_1_cofactor_second_place(self):
        label_model = LabelModel()
        label_model.add_compounds(["ATP", "ADP"])
        label_model.add_label_compounds({"x": 1, "y": 1})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "ATP": -1, "y": 1, "ADP": -1},
            labelmap=[0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0": {
                    "x__0": -1,
                    "ATP": -1,
                    "y__0": 1,
                    "ADP": -1,
                },
                "v1__1": {
                    "x__1": -1,
                    "ATP": -1,
                    "y__1": 1,
                    "ADP": -1,
                },
            },
        )

    def test_add_labelmap_reaction_2_2(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__00": {"x__00": -1, "y__00": 1},
                "v1__01": {"x__01": -1, "y__01": 1},
                "v1__10": {"x__10": -1, "y__10": 1},
                "v1__11": {"x__11": -1, "y__11": 1},
            },
        )

    def test_add_labelmap_reaction_2_2_reverse(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[1, 0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__00": {"x__00": -1, "y__00": 1},
                "v1__01": {"x__01": -1, "y__10": 1},
                "v1__10": {"x__10": -1, "y__01": 1},
                "v1__11": {"x__11": -1, "y__11": 1},
            },
        )

    def test_add_labelmap_substrate_stoich_two(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 4})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -2, "y": 1},
            labelmap=[0, 1, 2, 3],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0000": {"x__00": -2, "y__0000": 1},
                "v1__0001": {"x__00": -1, "x__01": -1, "y__0001": 1},
                "v1__0010": {"x__00": -1, "x__10": -1, "y__0010": 1},
                "v1__0011": {"x__00": -1, "x__11": -1, "y__0011": 1},
                "v1__0100": {"x__01": -1, "x__00": -1, "y__0100": 1},
                "v1__0101": {"x__01": -2, "y__0101": 1},
                "v1__0110": {"x__01": -1, "x__10": -1, "y__0110": 1},
                "v1__0111": {"x__01": -1, "x__11": -1, "y__0111": 1},
                "v1__1000": {"x__10": -1, "x__00": -1, "y__1000": 1},
                "v1__1001": {"x__10": -1, "x__01": -1, "y__1001": 1},
                "v1__1010": {"x__10": -2, "y__1010": 1},
                "v1__1011": {"x__10": -1, "x__11": -1, "y__1011": 1},
                "v1__1100": {"x__11": -1, "x__00": -1, "y__1100": 1},
                "v1__1101": {"x__11": -1, "x__01": -1, "y__1101": 1},
                "v1__1110": {"x__11": -1, "x__10": -1, "y__1110": 1},
                "v1__1111": {"x__11": -2, "y__1111": 1},
            },
        )

    def test_add_labelmap_product_stoich_two(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 4, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 2},
            labelmap=[0, 1, 2, 3],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0000": {"x__0000": -1, "y__00": 2},
                "v1__0001": {"x__0001": -1, "y__00": 1, "y__01": 1},
                "v1__0010": {"x__0010": -1, "y__00": 1, "y__10": 1},
                "v1__0011": {"x__0011": -1, "y__00": 1, "y__11": 1},
                "v1__0100": {"x__0100": -1, "y__01": 1, "y__00": 1},
                "v1__0101": {"x__0101": -1, "y__01": 2},
                "v1__0110": {"x__0110": -1, "y__01": 1, "y__10": 1},
                "v1__0111": {"x__0111": -1, "y__01": 1, "y__11": 1},
                "v1__1000": {"x__1000": -1, "y__10": 1, "y__00": 1},
                "v1__1001": {"x__1001": -1, "y__10": 1, "y__01": 1},
                "v1__1010": {"x__1010": -1, "y__10": 2},
                "v1__1011": {"x__1011": -1, "y__10": 1, "y__11": 1},
                "v1__1100": {"x__1100": -1, "y__11": 1, "y__00": 1},
                "v1__1101": {"x__1101": -1, "y__11": 1, "y__01": 1},
                "v1__1110": {"x__1110": -1, "y__11": 1, "y__10": 1},
                "v1__1111": {"x__1111": -1, "y__11": 2},
            },
        )

    def test_add_labelmap_product_two_substrates(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x1": 2, "x2": 2, "y": 4})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x1": -1, "x2": -1, "y": 1},
            labelmap=[0, 1, 2, 3],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0000": {"x1__00": -1, "x2__00": -1, "y__0000": 1},
                "v1__0001": {"x1__00": -1, "x2__01": -1, "y__0001": 1},
                "v1__0010": {"x1__00": -1, "x2__10": -1, "y__0010": 1},
                "v1__0011": {"x1__00": -1, "x2__11": -1, "y__0011": 1},
                "v1__0100": {"x1__01": -1, "x2__00": -1, "y__0100": 1},
                "v1__0101": {"x1__01": -1, "x2__01": -1, "y__0101": 1},
                "v1__0110": {"x1__01": -1, "x2__10": -1, "y__0110": 1},
                "v1__0111": {"x1__01": -1, "x2__11": -1, "y__0111": 1},
                "v1__1000": {"x1__10": -1, "x2__00": -1, "y__1000": 1},
                "v1__1001": {"x1__10": -1, "x2__01": -1, "y__1001": 1},
                "v1__1010": {"x1__10": -1, "x2__10": -1, "y__1010": 1},
                "v1__1011": {"x1__10": -1, "x2__11": -1, "y__1011": 1},
                "v1__1100": {"x1__11": -1, "x2__00": -1, "y__1100": 1},
                "v1__1101": {"x1__11": -1, "x2__01": -1, "y__1101": 1},
                "v1__1110": {"x1__11": -1, "x2__10": -1, "y__1110": 1},
                "v1__1111": {"x1__11": -1, "x2__11": -1, "y__1111": 1},
            },
        )

    def test_add_labelmap_product_two_substrates_reverse(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x1": 2, "x2": 2, "y": 4})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x1": -1, "x2": -1, "y": 1},
            labelmap=[3, 2, 1, 0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0000": {"x1__00": -1, "x2__00": -1, "y__0000": 1},
                "v1__0001": {"x1__00": -1, "x2__01": -1, "y__1000": 1},
                "v1__0010": {"x1__00": -1, "x2__10": -1, "y__0100": 1},
                "v1__0011": {"x1__00": -1, "x2__11": -1, "y__1100": 1},
                "v1__0100": {"x1__01": -1, "x2__00": -1, "y__0010": 1},
                "v1__0101": {"x1__01": -1, "x2__01": -1, "y__1010": 1},
                "v1__0110": {"x1__01": -1, "x2__10": -1, "y__0110": 1},
                "v1__0111": {"x1__01": -1, "x2__11": -1, "y__1110": 1},
                "v1__1000": {"x1__10": -1, "x2__00": -1, "y__0001": 1},
                "v1__1001": {"x1__10": -1, "x2__01": -1, "y__1001": 1},
                "v1__1010": {"x1__10": -1, "x2__10": -1, "y__0101": 1},
                "v1__1011": {"x1__10": -1, "x2__11": -1, "y__1101": 1},
                "v1__1100": {"x1__11": -1, "x2__00": -1, "y__0011": 1},
                "v1__1101": {"x1__11": -1, "x2__01": -1, "y__1011": 1},
                "v1__1110": {"x1__11": -1, "x2__10": -1, "y__0111": 1},
                "v1__1111": {"x1__11": -1, "x2__11": -1, "y__1111": 1},
            },
        )

    def test_add_labelmap_product_two_substrates_two_products(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x1": 2, "x2": 2, "y1": 2, "y2": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x1": -1, "x2": -1, "y1": 1, "y2": 1},
            labelmap=[0, 1, 2, 3],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0000": {"x1__00": -1, "x2__00": -1, "y1__00": 1, "y2__00": 1},
                "v1__0001": {"x1__00": -1, "x2__01": -1, "y1__00": 1, "y2__01": 1},
                "v1__0010": {"x1__00": -1, "x2__10": -1, "y1__00": 1, "y2__10": 1},
                "v1__0011": {"x1__00": -1, "x2__11": -1, "y1__00": 1, "y2__11": 1},
                "v1__0100": {"x1__01": -1, "x2__00": -1, "y1__01": 1, "y2__00": 1},
                "v1__0101": {"x1__01": -1, "x2__01": -1, "y1__01": 1, "y2__01": 1},
                "v1__0110": {"x1__01": -1, "x2__10": -1, "y1__01": 1, "y2__10": 1},
                "v1__0111": {"x1__01": -1, "x2__11": -1, "y1__01": 1, "y2__11": 1},
                "v1__1000": {"x1__10": -1, "x2__00": -1, "y1__10": 1, "y2__00": 1},
                "v1__1001": {"x1__10": -1, "x2__01": -1, "y1__10": 1, "y2__01": 1},
                "v1__1010": {"x1__10": -1, "x2__10": -1, "y1__10": 1, "y2__10": 1},
                "v1__1011": {"x1__10": -1, "x2__11": -1, "y1__10": 1, "y2__11": 1},
                "v1__1100": {"x1__11": -1, "x2__00": -1, "y1__11": 1, "y2__00": 1},
                "v1__1101": {"x1__11": -1, "x2__01": -1, "y1__11": 1, "y2__01": 1},
                "v1__1110": {"x1__11": -1, "x2__10": -1, "y1__11": 1, "y2__10": 1},
                "v1__1111": {"x1__11": -1, "x2__11": -1, "y1__11": 1, "y2__11": 1},
            },
        )

    def test_add_labelmap_product_two_substrates_two_products_reverse(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x1": 2, "x2": 2, "y1": 2, "y2": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x1": -1, "x2": -1, "y1": 1, "y2": 1},
            labelmap=[3, 2, 1, 0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__0000": {"x1__00": -1, "x2__00": -1, "y1__00": 1, "y2__00": 1},
                "v1__0001": {"x1__00": -1, "x2__01": -1, "y1__10": 1, "y2__00": 1},
                "v1__0010": {"x1__00": -1, "x2__10": -1, "y1__01": 1, "y2__00": 1},
                "v1__0011": {"x1__00": -1, "x2__11": -1, "y1__11": 1, "y2__00": 1},
                "v1__0100": {"x1__01": -1, "x2__00": -1, "y1__00": 1, "y2__10": 1},
                "v1__0101": {"x1__01": -1, "x2__01": -1, "y1__10": 1, "y2__10": 1},
                "v1__0110": {"x1__01": -1, "x2__10": -1, "y1__01": 1, "y2__10": 1},
                "v1__0111": {"x1__01": -1, "x2__11": -1, "y1__11": 1, "y2__10": 1},
                "v1__1000": {"x1__10": -1, "x2__00": -1, "y1__00": 1, "y2__01": 1},
                "v1__1001": {"x1__10": -1, "x2__01": -1, "y1__10": 1, "y2__01": 1},
                "v1__1010": {"x1__10": -1, "x2__10": -1, "y1__01": 1, "y2__01": 1},
                "v1__1011": {"x1__10": -1, "x2__11": -1, "y1__11": 1, "y2__01": 1},
                "v1__1100": {"x1__11": -1, "x2__00": -1, "y1__00": 1, "y2__11": 1},
                "v1__1101": {"x1__11": -1, "x2__01": -1, "y1__10": 1, "y2__11": 1},
                "v1__1110": {"x1__11": -1, "x2__10": -1, "y1__01": 1, "y2__11": 1},
                "v1__1111": {"x1__11": -1, "x2__11": -1, "y1__11": 1, "y2__11": 1},
            },
        )

    def test_add_labelmap_reaction_warns_on_no_external(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 3})
        with self.assertWarns(UserWarning):
            label_model.add_labelmap_reaction(
                rate_name="v1",
                function=lambda *args: 0,
                stoichiometry={"x": -1, "y": 1},
                labelmap=[0, 1],
            )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__001": {"x__00": -1, "y__00": 1},
                "v1__011": {"x__01": -1, "y__01": 1},
                "v1__101": {"x__10": -1, "y__10": 1},
                "v1__111": {"x__11": -1, "y__11": 1},
            },
        )

    def test_add_labelmap_reaction_missing_labels(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 3})
        with self.assertRaises(ValueError):
            label_model.add_labelmap_reaction(
                rate_name="v1",
                function=lambda *args: 0,
                stoichiometry={"x": -1, "y": 1},
                labelmap=[0],
            )

    def test_add_labelmap_reaction_external_labels(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 3})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
            external_labels=[0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__001": {"x__00": -1, "y__00": 1},
                "v1__011": {"x__01": -1, "y__01": 1},
                "v1__101": {"x__10": -1, "y__10": 1},
                "v1__111": {"x__11": -1, "y__11": 1},
            },
        )

    def test_add_labelmap_reaction_missing_substrate(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2})
        with self.assertRaises(KeyError):
            label_model.add_labelmap_reaction(
                rate_name="v1",
                function=lambda *args: 0,
                stoichiometry={"x": -1, "y": 1},
                labelmap=[0, 1],
                external_labels=[0],
            )

    def test_update_labelmap(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.update_labelmap_reaction(
            rate_name="v1",
            labelmap=[1, 0],
        )
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__00": {"x__00": -1, "y__00": 1},
                "v1__01": {"x__01": -1, "y__10": 1},
                "v1__10": {"x__10": -1, "y__01": 1},
                "v1__11": {"x__11": -1, "y__11": 1},
            },
        )

    def test_update_labelmap_old(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.update_labelmap_reaction(rate_name="v1")
        self.assertEqual(
            label_model.stoichiometries,
            {
                "v1__00": {"x__00": -1, "y__00": 1},
                "v1__01": {"x__01": -1, "y__01": 1},
                "v1__10": {"x__10": -1, "y__10": 1},
                "v1__11": {"x__11": -1, "y__11": 1},
            },
        )

    def test_remove_labelmap_reaction(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.remove_labelmap_reaction(rate_name="v1")
        self.assertEqual(label_model.rates, {})
        self.assertEqual(label_model.stoichiometries, {})
        self.assertEqual(label_model.stoichiometries_by_compounds, {})
        self.assertEqual(label_model.base_reactions, {})
        self.assertEqual(label_model.meta_info["rates"], {})

    def test_remove_labelmap_reactions(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.add_labelmap_reaction(
            rate_name="v2",
            function=lambda *args: 0,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.remove_labelmap_reactions(rate_names=["v1", "v2"])
        self.assertEqual(label_model.rates, {})
        self.assertEqual(label_model.stoichiometries, {})
        self.assertEqual(label_model.stoichiometries_by_compounds, {})
        self.assertEqual(label_model.base_reactions, {})
        self.assertEqual(label_model.meta_info["rates"], {})


class GenerateY0Tests(unittest.TestCase):
    def create_model_and_y0(self):
        m = LabelModel()
        m.add_label_compound(compound="GAP", num_labels=3)
        base_y0 = {"GAP": 1}
        return m, base_y0.copy()

    @staticmethod
    def reduce_y0(y0):
        return {k: v for k, v in y0.items() if v != 0}

    def test_generate_y0_no_label(self):
        m = LabelModel()
        m.add_compound("x")
        y0 = m.generate_y0({"x": 1})
        self.assertEqual(y0, {"x": 1})

    def test_generate_y0_empty(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0)
        self.assertEqual(self.reduce_y0(y0), {"GAP__000": 1})

    def test_generate_y0_non_dict(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=[1])
        self.assertEqual(self.reduce_y0(y0), {"GAP__000": 1})

    def test_generate_y0_0(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": 0})
        self.assertEqual(self.reduce_y0(y0), {"GAP__100": 1})

    def test_generate_y0_1(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": 1})
        self.assertEqual(self.reduce_y0(y0), {"GAP__010": 1})

    def test_generate_y0_2(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": 2})
        self.assertEqual(self.reduce_y0(y0), {"GAP__001": 1})

    def test_generate_y0_0_1(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": (0, 1)})
        self.assertEqual(self.reduce_y0(y0), {"GAP__110": 1})

    def test_generate_y0_0_2(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": (0, 2)})
        self.assertEqual(self.reduce_y0(y0), {"GAP__101": 1})

    def test_generate_y0_1_2(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": (1, 2)})
        self.assertEqual(self.reduce_y0(y0), {"GAP__011": 1})

    def test_generate_y0_0_1_2(self):
        m, base_y0 = self.create_model_and_y0()
        y0 = m.generate_y0(base_y0=base_y0, label_positions={"GAP": (0, 1, 2)})
        self.assertEqual(self.reduce_y0(y0), {"GAP__111": 1})


class ModelConversionTests(unittest.TestCase):
    def test_to_model(self):
        label_model = LabelModel()
        label_model.add_compounds(("x", "y"))
        label_model.add_label_compounds({"a": 2, "b": 2})
        label_model.add_algebraic_module(
            module_name="mod1",
            function=lambda *args: args,
            compounds=["x", "a"],
            derived_compounds=["c"],
            modifiers=["y", "b"],
            parameters=None,
        )
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda *args: 1,
            stoichiometry={"x": -1, "a": -1, "y": 1, "b": 1},
            labelmap=[0, 1],
            external_labels=None,
            modifiers=["x", "a", "y", "b__0"],
            parameters=None,
            reversible=True,
        )
        m = label_model.to_model()
        self.assertEqual(m.compounds, ["a", "b", "x", "y"])
        self.assertEqual(m.get_derived_compounds(), ["c"])

        mod = m.algebraic_modules["mod1"]
        self.assertEqual(mod["compounds"], ["x", "a"])
        self.assertEqual(mod["derived_compounds"], ["c"])
        self.assertEqual(mod["modifiers"], ["y", "b"])
        self.assertEqual(mod["parameters"], [])

        rate = m.rates["v1"]
        self.assertEqual(rate["substrates"], ["x", "a"])
        self.assertEqual(rate["products"], ["y", "b"])
        self.assertEqual(rate["modifiers"], ["x", "a", "y", "b"])
        self.assertEqual(
            rate["dynamic_variables"], ["x", "a", "y", "b", "x", "a", "y", "b"]
        )


class SimulationFunctionTests(unittest.TestCase):
    def test_get_total_fluxes(self):
        label_model = LabelModel()
        label_model.add_label_compounds({"x": 2, "y": 2})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda x: 1,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.add_labelmap_reaction(
            rate_name="v2",
            function=lambda x: 1,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )

        y = label_model.generate_y0({"x": 0, "y": 0})
        self.assertEqual(label_model.get_total_fluxes("v1", y, t=0), 4)


class LabelScopeTests(unittest.TestCase):
    def test_label_scope_forward(self):
        label_model = LabelModel()
        label_model.add_compound("ATP")
        label_model.add_label_compounds({"x": 2, "y": 2, "z": 2, "ADP": 0})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda x: 1,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.add_labelmap_reaction(
            rate_name="v2",
            function=lambda x: 1,
            stoichiometry={"y": -1, "z": 1},
            labelmap=[1, 0],
        )
        self.assertEqual(
            label_model.get_label_scope({"x": 0}), {0: {"y__10"}, 1: {"z__01"}}
        )

    def test_label_scope_backward(self):
        label_model = LabelModel()
        label_model.add_compound("ATP")
        label_model.add_label_compounds({"x": 2, "y": 2, "z": 2, "ADP": 0})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda x: 1,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
            reversible=True,
        )
        label_model.add_labelmap_reaction(
            rate_name="v2",
            function=lambda x: 1,
            stoichiometry={"y": -1, "z": 1},
            labelmap=[1, 0],
            reversible=True,
        )
        self.assertEqual(
            label_model.get_label_scope({"z": 1}), {0: {"y__10"}, 1: {"x__10"}}
        )

    def test_label_scope_backward_fail_without_reversible(self):
        label_model = LabelModel()
        label_model.add_compound("ATP")
        label_model.add_label_compounds({"x": 2, "y": 2, "z": 2, "ADP": 0})
        label_model.add_labelmap_reaction(
            rate_name="v1",
            function=lambda x: 1,
            stoichiometry={"x": -1, "y": 1},
            labelmap=[0, 1],
        )
        label_model.add_labelmap_reaction(
            rate_name="v2",
            function=lambda x: 1,
            stoichiometry={"y": -1, "z": 1},
            labelmap=[1, 0],
        )
        self.assertEqual(label_model.get_label_scope({"z": 1}), {})


class SBMLTests(unittest.TestCase):
    def test_create_sbml_compounds_no_meta_info(self):
        m = LabelModel()
        m.add_parameters({"k1": 1})
        m.add_compound("ATP")
        m.add_label_compounds({"X": 2, "Y": 2})

        doc = m._create_sbml_document()
        sbml_model = m._create_sbml_model(doc=doc)
        m._create_sbml_compounds(sbml_model=sbml_model)

        self.assertEqual(
            [i.getId() for i in sbml_model.getListOfSpecies()],
            [
                "X__00",
                "X__01",
                "X__10",
                "X__11",
                "Y__00",
                "Y__01",
                "Y__10",
                "Y__11",
                "ATP",
            ],
        )

        self.assertEqual(sbml_model.getSpecies("X__00").getName(), "")
        self.assertEqual(sbml_model.getSpecies("ATP").getName(), "")

    def test_create_sbml_compounds_meta_info(self):
        m = LabelModel()
        m.add_parameters({"k1": 1})
        m.add_compounds(
            ["ATP"],
            meta_info={
                "ATP": {
                    "common_name": "ATP",
                    "charge": -2.0,
                    "compartment": "e",
                    "formula": "C6H12O6",
                }
            },
        )
        m.add_label_compounds(
            {"X": 2, "Y": 2},
            meta_info={
                "X": {
                    "common_name": "Glucose",
                    "charge": -2.0,
                    "compartment": "e",
                    "formula": "C6H12O6",
                }
            },
        )

        doc = m._create_sbml_document()
        sbml_model = m._create_sbml_model(doc=doc)
        m._create_sbml_compounds(sbml_model=sbml_model)

        self.assertEqual(
            [i.getId() for i in sbml_model.getListOfSpecies()],
            [
                "X__00",
                "X__01",
                "X__10",
                "X__11",
                "Y__00",
                "Y__01",
                "Y__10",
                "Y__11",
                "ATP",
            ],
        )

        cpd = sbml_model.getSpecies("X__00")
        self.assertEqual(cpd.getId(), "X__00")
        self.assertEqual(cpd.getName(), "Glucose")
        self.assertEqual(cpd.getCompartment(), "e")
        self.assertEqual(cpd.getPlugin("fbc").getCharge(), -2)
        self.assertEqual(cpd.getPlugin("fbc").getChemicalFormula(), "C6H12O6")
        self.assertEqual(cpd.getConstant(), False)
        self.assertEqual(cpd.getBoundaryCondition(), False)

        cpd = sbml_model.getSpecies("ATP")
        self.assertEqual(cpd.getId(), "ATP")
        self.assertEqual(cpd.getName(), "ATP")
        self.assertEqual(cpd.getCompartment(), "e")
        self.assertEqual(cpd.getPlugin("fbc").getCharge(), -2)
        self.assertEqual(cpd.getPlugin("fbc").getChemicalFormula(), "C6H12O6")
        self.assertEqual(cpd.getConstant(), False)
        self.assertEqual(cpd.getBoundaryCondition(), False)

    def test_create_sbml_reactions_no_meta_info(self):
        m = LabelModel()
        m.add_parameters({"k1": 1})
        m.add_compounds(["ATP", "ADP"])
        m.add_label_compounds({"X": 2, "Y": 2})
        m.add_labelmap_reaction(
            rate_name="v1",
            function=rf.mass_action_1,
            labelmap=[0, 1],
            stoichiometry={"X": -1, "Y": 1},
            modifiers=["ATP", "ADP"],
            parameters=["k1"],
            reversible=False,
        )

        doc = m._create_sbml_document()
        sbml_model = m._create_sbml_model(doc=doc)
        m._create_sbml_reactions(sbml_model=sbml_model)

        self.assertEqual(
            [i.getId() for i in sbml_model.getListOfReactions()],
            ["v1__00", "v1__01", "v1__10", "v1__11"],
        )

        rxn = sbml_model.getReaction("v1__00")
        self.assertEqual(rxn.getReversible(), False)
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "X__00")
        self.assertEqual(rxn.getListOfReactants()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfReactants()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "Y__00")
        self.assertEqual(rxn.getListOfProducts()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfProducts()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfModifiers()[0].getSpecies(), "ATP")
        self.assertEqual(rxn.getListOfModifiers()[1].getSpecies(), "ADP")
        self.assertEqual(rxn.getKineticLaw(), None)

        rxn = sbml_model.getReaction("v1__11")
        self.assertEqual(rxn.getReversible(), False)
        self.assertEqual(rxn.getListOfReactants()[0].getSpecies(), "X__11")
        self.assertEqual(rxn.getListOfReactants()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfReactants()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfProducts()[0].getSpecies(), "Y__11")
        self.assertEqual(rxn.getListOfProducts()[0].getStoichiometry(), 1.0)
        self.assertEqual(rxn.getListOfProducts()[0].getConstant(), False)
        self.assertEqual(rxn.getListOfModifiers()[0].getSpecies(), "ATP")
        self.assertEqual(rxn.getListOfModifiers()[1].getSpecies(), "ADP")
        self.assertEqual(rxn.getKineticLaw(), None)

    def test_create_sbml_reactions_meta_info(self):
        m = LabelModel()
        m.add_parameters({"k1": 1})
        m.add_compounds(["ATP", "ADP"])
        m.add_label_compounds({"X": 2, "Y": 2})
        m.add_labelmap_reaction(
            rate_name="v1",
            function=rf.mass_action_1,
            labelmap=[0, 1],
            stoichiometry={"X": -1, "Y": 1},
            modifiers=["ATP", "ADP"],
            parameters=["k1"],
            reversible=False,
            **{
                "common_name": "test",
                "gibbs0": 1,
                "ec": "123",
                "database_links": {"metacyc": "test"},
                "notes": {"test": "test"},
                "sbml_function": None,
            },
        )

        doc = m._create_sbml_document()
        sbml_model = m._create_sbml_model(doc=doc)
        m._create_sbml_reactions(sbml_model=sbml_model)

        rxn = sbml_model.getReaction("v1__00")
        self.assertEqual(rxn.getName(), "test")
