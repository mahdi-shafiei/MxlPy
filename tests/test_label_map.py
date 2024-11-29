from __future__ import annotations

from modelbase2.label_map import (
    _assign_compound_labels,
    _generate_binary_labels,
    _get_external_labels,
    _get_labels_per_variable,
    _map_substrates_to_products,
    _repack_stoichiometries,
    _split_label_string,
    _unpack_stoichiometries,
)


def test_generate_binary_labels_no_labels() -> None:
    result = _generate_binary_labels(
        base_name="cpd",
        num_labels=0,
    )
    assert result == ["cpd"]


def test_generate_binary_labels_one_label() -> None:
    result = _generate_binary_labels(
        base_name="cpd",
        num_labels=1,
    )
    assert result == ["cpd__0", "cpd__1"]


def test_generate_binary_labels_two_labels() -> None:
    result = _generate_binary_labels(
        base_name="cpd",
        num_labels=2,
    )
    assert result == ["cpd__00", "cpd__01", "cpd__10", "cpd__11"]


def test_generate_binary_labels_three_labels() -> None:
    result = _generate_binary_labels(
        base_name="cpd",
        num_labels=3,
    )
    assert result == [
        "cpd__000",
        "cpd__001",
        "cpd__010",
        "cpd__011",
        "cpd__100",
        "cpd__101",
        "cpd__110",
        "cpd__111",
    ]


def test_split_label_string_single_compound() -> None:
    result = _split_label_string(
        label="01",
        labels_per_compound=[2],
    )
    assert result == ["01"]


def test_split_label_string_two_compounds() -> None:
    result = _split_label_string(
        label="01",
        labels_per_compound=[1, 1],
    )
    assert result == ["0", "1"]


def test_split_label_string_single_compound_four_labels() -> None:
    result = _split_label_string(
        label="0011",
        labels_per_compound=[4],
    )
    assert result == ["0011"]


def test_split_label_string_three_and_one_labels() -> None:
    result = _split_label_string(
        label="0011",
        labels_per_compound=[3, 1],
    )
    assert result == ["001", "1"]


def test_split_label_string_two_and_two_labels() -> None:
    result = _split_label_string(
        label="0011",
        labels_per_compound=[2, 2],
    )
    assert result == ["00", "11"]


def test_split_label_string_one_and_three_labels() -> None:
    result = _split_label_string(
        label="0011",
        labels_per_compound=[1, 3],
    )
    assert result == ["0", "011"]


def test_map_substrates_to_products_simple() -> None:
    result = _map_substrates_to_products(
        rate_suffix="01",
        labelmap=[1, 0],
    )
    assert result == "10"


def test_map_substrates_to_products_identity() -> None:
    result = _map_substrates_to_products(
        rate_suffix="01",
        labelmap=[0, 1],
    )
    assert result == "01"


def test_map_substrates_to_products_reversed() -> None:
    result = _map_substrates_to_products(
        rate_suffix="1100",
        labelmap=[3, 2, 1, 0],
    )
    assert result == "0011"


def test_map_substrates_to_products_duplicate() -> None:
    result = _map_substrates_to_products(
        rate_suffix="1010",
        labelmap=[0, 0, 1, 1],
    )
    assert result == "1100"


def test_map_substrates_to_products_empty() -> None:
    result = _map_substrates_to_products(
        rate_suffix="",
        labelmap=[],
    )
    assert result == ""


def test_unpack_stoichiometries_single_substrate_single_product() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={"A": -1, "B": 1},
    )
    assert result == (["A"], ["B"])


def test_unpack_stoichiometries_multiple_substrates_single_product() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={"A": -2, "B": 1},
    )
    assert result == (["A", "A"], ["B"])


def test_unpack_stoichiometries_single_substrate_multiple_products() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={"A": -1, "B": 2},
    )
    assert result == (["A"], ["B", "B"])


def test_unpack_stoichiometries_multiple_substrates_multiple_products() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={"A": -2, "B": 2},
    )
    assert result == (["A", "A"], ["B", "B"])


def test_unpack_stoichiometries_no_substrates() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={"A": 1},
    )
    assert result == ([], ["A"])


def test_unpack_stoichiometries_no_products() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={"A": -1},
    )
    assert result == (["A"], [])


def test_unpack_stoichiometries_empty() -> None:
    result = _unpack_stoichiometries(
        stoichiometries={},
    )
    assert result == ([], [])


def test_get_labels_per_variable_all_labeled() -> None:
    result = _get_labels_per_variable(
        label_variables={"A": 2, "B": 3},
        compounds=["A", "B"],
    )
    assert result == [2, 3]


def test_get_labels_per_variable_some_labeled() -> None:
    result = _get_labels_per_variable(
        label_variables={"A": 2, "B": 3},
        compounds=["A", "C"],
    )
    assert result == [2, 0]


def test_get_labels_per_variable_none_labeled() -> None:
    result = _get_labels_per_variable(
        label_variables={"A": 2, "B": 3},
        compounds=["C", "D"],
    )
    assert result == [0, 0]


def test_get_labels_per_variable_empty_compounds() -> None:
    result = _get_labels_per_variable(
        label_variables={"A": 2, "B": 3},
        compounds=[],
    )
    assert result == []


def test_get_labels_per_variable_empty_label_variables() -> None:
    result = _get_labels_per_variable(
        label_variables={},
        compounds=["A", "B"],
    )
    assert result == [0, 0]


def test_repack_stoichiometries_single_substrate_single_product() -> None:
    result = _repack_stoichiometries(
        new_substrates=["A"],
        new_products=["B"],
    )
    assert result == {"A": -1, "B": 1}


def test_repack_stoichiometries_multiple_substrates_single_product() -> None:
    result = _repack_stoichiometries(
        new_substrates=["A", "A"],
        new_products=["B"],
    )
    assert result == {"A": -2, "B": 1}


def test_repack_stoichiometries_single_substrate_multiple_products() -> None:
    result = _repack_stoichiometries(
        new_substrates=["A"],
        new_products=["B", "B"],
    )
    assert result == {"A": -1, "B": 2}


def test_repack_stoichiometries_multiple_substrates_multiple_products() -> None:
    result = _repack_stoichiometries(
        new_substrates=["A", "A"],
        new_products=["B", "B"],
    )
    assert result == {"A": -2, "B": 2}


def test_repack_stoichiometries_no_substrates() -> None:
    result = _repack_stoichiometries(
        new_substrates=[],
        new_products=["A"],
    )
    assert result == {"A": 1}


def test_repack_stoichiometries_no_products() -> None:
    result = _repack_stoichiometries(
        new_substrates=["A"],
        new_products=[],
    )
    assert result == {"A": -1}


def test_repack_stoichiometries_empty() -> None:
    result = _repack_stoichiometries(
        new_substrates=[],
        new_products=[],
    )
    assert result == {}


def test_assign_compound_labels_no_suffixes() -> None:
    result = _assign_compound_labels(
        base_compounds=["A", "B"],
        label_suffixes=["", ""],
    )
    assert result == ["A", "B"]


def test_assign_compound_labels_with_suffixes() -> None:
    result = _assign_compound_labels(
        base_compounds=["A", "B"],
        label_suffixes=["01", "10"],
    )
    assert result == ["A__01", "B__10"]


def test_assign_compound_labels_mixed_suffixes() -> None:
    result = _assign_compound_labels(
        base_compounds=["A", "B"],
        label_suffixes=["", "10"],
    )
    assert result == ["A", "B__10"]


def test_assign_compound_labels_single_compound() -> None:
    result = _assign_compound_labels(
        base_compounds=["A"],
        label_suffixes=["01"],
    )
    assert result == ["A__01"]


def test_assign_compound_labels_empty_lists() -> None:
    result = _assign_compound_labels(
        base_compounds=[],
        label_suffixes=[],
    )
    assert result == []


def test_get_external_labels_no_external_labels() -> None:
    result = _get_external_labels(
        total_product_labels=2,
        total_substrate_labels=2,
    )
    assert result == ""


def test_get_external_labels_one_external_label() -> None:
    result = _get_external_labels(
        total_product_labels=3,
        total_substrate_labels=2,
    )
    assert result == "1"


def test_get_external_labels_multiple_external_labels() -> None:
    result = _get_external_labels(
        total_product_labels=5,
        total_substrate_labels=2,
    )
    assert result == "111"


def test_get_external_labels_negative_external_labels() -> None:
    result = _get_external_labels(
        total_product_labels=2,
        total_substrate_labels=3,
    )
    assert result == ""


def test_get_external_labels_zero_labels() -> None:
    result = _get_external_labels(
        total_product_labels=0,
        total_substrate_labels=0,
    )
    assert result == ""
