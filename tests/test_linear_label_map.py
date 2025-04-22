from __future__ import annotations

import pytest

from modelbase2.linear_label_map import (
    Derived,
    _add_label_influx_or_efflux,
    _generate_isotope_labels,
    _map_substrates_to_labelmap,
    _stoichiometry_to_duplicate_list,
    _unpack_stoichiometries,
)


def test_generate_isotope_labels_valid() -> None:
    result = _generate_isotope_labels("x", 2)
    assert result == ["x__0", "x__1"]


def test_generate_isotope_labels_zero_labels() -> None:
    with pytest.raises(ValueError, match="Compound x must have labels"):
        _generate_isotope_labels("x", 0)


def test_generate_isotope_labels_negative_labels() -> None:
    with pytest.raises(ValueError, match="Compound x must have labels"):
        _generate_isotope_labels("x", -1)


def test_unpack_stoichiometries_valid() -> None:
    stoichiometries = {"A": -1.0, "B": 2.0}
    substrates, products = _unpack_stoichiometries(stoichiometries)
    assert substrates == {"A": 1}
    assert products == {"B": 2}


def test_unpack_stoichiometries_with_derived() -> None:
    stoichiometries = {"A": Derived(name="A", fn=lambda x: x, args=[]), "B": 2.0}
    with pytest.raises(NotImplementedError):
        _unpack_stoichiometries(stoichiometries)


def test_stoichiometry_to_duplicate_list_valid() -> None:
    stoichiometry = {"A": 2, "B": 1}
    result = _stoichiometry_to_duplicate_list(stoichiometry)
    assert result == ["A", "A", "B"]


def test_stoichiometry_to_duplicate_list_empty() -> None:
    stoichiometry = {}
    result = _stoichiometry_to_duplicate_list(stoichiometry)
    assert result == []


def test_stoichiometry_to_duplicate_list_single_entry() -> None:
    stoichiometry = {"A": 3}
    result = _stoichiometry_to_duplicate_list(stoichiometry)
    assert result == ["A", "A", "A"]


def test_map_substrates_to_labelmap_valid() -> None:
    substrates = ["A", "B"]
    labelmap = [1, 0]
    result = _map_substrates_to_labelmap(substrates, labelmap)
    assert result == ["B", "A"]


def test_map_substrates_to_labelmap_empty() -> None:
    substrates = []
    labelmap = []
    result = _map_substrates_to_labelmap(substrates, labelmap)
    assert result == []


def test_map_substrates_to_labelmap_single_entry() -> None:
    substrates = ["A"]
    labelmap = [0]
    result = _map_substrates_to_labelmap(substrates, labelmap)
    assert result == ["A"]


def test_map_substrates_to_labelmap_mismatched_lengths() -> None:
    substrates = ["A", "B"]
    labelmap = [1]
    with pytest.raises(ValueError):
        _map_substrates_to_labelmap(substrates, labelmap)


def test_add_label_influx_or_efflux_balanced() -> None:
    substrates = ["A", "B"]
    products = ["C", "D"]
    labelmap = [0, 1]
    result_substrates, result_products = _add_label_influx_or_efflux(
        substrates, products, labelmap
    )
    assert result_substrates == ["A", "B"]
    assert result_products == ["C", "D"]


def test_add_label_influx_or_efflux_more_substrates() -> None:
    substrates = ["A", "B", "C"]
    products = ["D"]
    labelmap = [0, 1, 2]
    result_substrates, result_products = _add_label_influx_or_efflux(
        substrates, products, labelmap
    )
    assert result_substrates == ["A", "B", "C"]
    assert result_products == ["D", "EXT", "EXT"]


def test_add_label_influx_or_efflux_more_products() -> None:
    substrates = ["A"]
    products = ["B", "C", "D"]
    labelmap = [0, 1, 2]
    result_substrates, result_products = _add_label_influx_or_efflux(
        substrates, products, labelmap
    )
    assert result_substrates == ["A", "EXT", "EXT"]
    assert result_products == ["B", "C", "D"]


def test_add_label_influx_or_efflux_labelmap_too_short() -> None:
    substrates = ["A", "B"]
    products = ["C"]
    labelmap = [0]
    with pytest.raises(ValueError, match="Labelmap 'missing' 1 label"):
        _add_label_influx_or_efflux(substrates, products, labelmap)
