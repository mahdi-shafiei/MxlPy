from __future__ import annotations

from modelbase2.experimental.codegen import (
    generate_model_code_py,
)
from tests import models

# def _generate_tests() -> None:
#     for i in sorted([i for i in dir(models) if i.startswith("m_")]):
#         source = generate_model_code_py(getattr(models, i)()).split("\n")
#         print(rf"""def test_generate_model_code_py_{i}() -> None:
#                 assert generate_model_code_py(models.{i}()).split('\n') == {source}
#             """)


def test_generate_model_code_py_m_0v_1p_0d_0r() -> None:
    assert generate_model_code_py(models.m_0v_1p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    p1 = 1.0",
        "    return ()",
    ]


def test_generate_model_code_py_m_0v_2p_0d_0r() -> None:
    assert generate_model_code_py(models.m_0v_2p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    return ()",
    ]


def test_generate_model_code_py_m_1v_0p_0d_0r() -> None:
    assert generate_model_code_py(models.m_1v_0p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1 = y",
        "    return dv1dt",
    ]


def test_generate_model_code_py_m_1v_1p_1d_0r() -> None:
    assert generate_model_code_py(models.m_1v_1p_1d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1 = y",
        "    p1 = 1.0",
        "    d1 = v1 + p1",
        "    return dv1dt",
    ]


def test_generate_model_code_py_m_1v_1p_1d_1r() -> None:
    assert generate_model_code_py(models.m_1v_1p_1d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1 = y",
        "    p1 = 1.0",
        "    d1 = v1 + p1",
        "    r1 = p1 * v1",
        "    dv1dt = - r1",
        "    return dv1dt",
    ]


def test_generate_model_code_py_m_2v_0p_0d_0r() -> None:
    assert generate_model_code_py(models.m_2v_0p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1, v2 = y",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_1p_1d_1r() -> None:
    assert generate_model_code_py(models.m_2v_1p_1d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1, v2 = y",
        "    p1 = 1.0",
        "    d1 = v1 + v2",
        "    r1 = p1 * v1",
        "    dv1dt = - r1",
        "    dv2dt = r1",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_2p_1d_1r() -> None:
    assert generate_model_code_py(models.m_2v_2p_1d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1, v2 = y",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    d1 = v1 + v2",
        "    r1 = p1 * v1",
        "    dv1dt = - r1",
        "    dv2dt = r1",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_2p_2d_1r() -> None:
    assert generate_model_code_py(models.m_2v_2p_2d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1, v2 = y",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    d1 = v1 + v2",
        "    d2 = v1 * v2",
        "    r1 = p1 * v1",
        "    dv1dt = - r1",
        "    dv2dt = r1",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_2p_2d_2r() -> None:
    assert generate_model_code_py(models.m_2v_2p_2d_2r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1, v2 = y",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    d1 = v1 + v2",
        "    d2 = v1 * v2",
        "    r1 = p1 * v1",
        "    r2 = p2 * v2",
        "    dv1dt = - r1 + r2",
        "    dv2dt = r1 - r2",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_dependent_derived() -> None:
    assert generate_model_code_py(models.m_dependent_derived()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    p1 = 1.0",
        "    d1 = p1",
        "    d2 = d1",
        "    return ()",
    ]


def test_generate_model_code_py_m_derived_stoichiometry() -> None:
    assert generate_model_code_py(models.m_derived_stoichiometry()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from modelbase2.types import Float",
        "",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        "    v1 = y",
        "    r1 = v1",
        "    dv1dt = 1.0 / v1 * r1",
        "    return dv1dt",
    ]
