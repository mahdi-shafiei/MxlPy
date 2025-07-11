from __future__ import annotations

from mxlpy.meta import generate_mxlpy_code
from tests import models


def test_generate_model_code_py_m_1v_0p_0d_0r() -> None:
    assert generate_mxlpy_code(models.m_1v_0p_0d_0r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "    )",
    ]


def test_generate_model_code_py_m_2v_0p_0d_0r() -> None:
    assert generate_mxlpy_code(models.m_2v_0p_0d_0r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_variable('v2', initial_value=2.0)",
        "    )",
    ]


def test_generate_model_code_py_m_0v_1p_0d_0r() -> None:
    assert generate_mxlpy_code(models.m_0v_1p_0d_0r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameter('p1', value=1.0)",
        "    )",
    ]


def test_generate_model_code_py_m_0v_2p_0d_0r() -> None:
    assert generate_mxlpy_code(models.m_0v_2p_0d_0r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameter('p1', value=1.0)",
        "        .add_parameter('p2', value=2.0)",
        "    )",
    ]


def test_generate_model_code_py_m_1v_1p_1d_0r() -> None:
    assert generate_mxlpy_code(models.m_1v_1p_1d_0r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def add(v1: float, p1: float) -> float:",
        "    return p1 + v1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_parameter('p1', value=1.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=add,",
        "                args=['v1', 'p1'],",
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_1v_1p_1d_1r() -> None:
    assert generate_mxlpy_code(models.m_1v_1p_1d_1r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def add(v1: float, p1: float) -> float:",
        "    return p1 + v1",
        "    ",
        "",
        "def mass_action_1s(v1: float, d1: float) -> float:",
        "    return d1*v1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_parameter('p1', value=1.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=add,",
        "                args=['v1', 'p1'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'd1'],",
        '                stoichiometry={"v1": -1.0},',
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_2v_1p_1d_1r() -> None:
    assert generate_mxlpy_code(models.m_2v_1p_1d_1r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def add(v1: float, v2: float) -> float:",
        "    return v1 + v2",
        "    ",
        "",
        "def mass_action_1s(v1: float, p1: float) -> float:",
        "    return p1*v1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_variable('v2', initial_value=2.0)",
        "        .add_parameter('p1', value=1.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=add,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'p1'],",
        '                stoichiometry={"v1": -1.0,"v2": 1.0},',
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_2v_2p_1d_1r() -> None:
    assert generate_mxlpy_code(models.m_2v_2p_1d_1r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def add(v1: float, v2: float) -> float:",
        "    return v1 + v2",
        "    ",
        "",
        "def mass_action_1s(v1: float, p1: float) -> float:",
        "    return p1*v1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_variable('v2', initial_value=2.0)",
        "        .add_parameter('p1', value=1.0)",
        "        .add_parameter('p2', value=2.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=add,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'p1'],",
        '                stoichiometry={"v1": -1.0,"v2": 1.0},',
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_2v_2p_2d_1r() -> None:
    assert generate_mxlpy_code(models.m_2v_2p_2d_1r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def add(v1: float, v2: float) -> float:",
        "    return v1 + v2",
        "    ",
        "",
        "def mul(v1: float, v2: float) -> float:",
        "    return v1*v2",
        "    ",
        "",
        "def mass_action_1s(v1: float, p1: float) -> float:",
        "    return p1*v1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_variable('v2', initial_value=2.0)",
        "        .add_parameter('p1', value=1.0)",
        "        .add_parameter('p2', value=2.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=add,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_derived(",
        "                'd2',",
        "                fn=mul,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'p1'],",
        '                stoichiometry={"v1": -1.0,"v2": 1.0},',
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_2v_2p_2d_2r() -> None:
    assert generate_mxlpy_code(models.m_2v_2p_2d_2r()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def add(v1: float, p1: float) -> float:",
        "    return p1 + v1",
        "    ",
        "",
        "def mul(v2: float, p2: float) -> float:",
        "    return p2*v2",
        "    ",
        "",
        "def mass_action_1s(v2: float, d2: float) -> float:",
        "    return d2*v2",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_variable('v2', initial_value=2.0)",
        "        .add_parameter('p1', value=1.0)",
        "        .add_parameter('p2', value=2.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=add,",
        "                args=['v1', 'p1'],",
        "            )",
        "        .add_derived(",
        "                'd2',",
        "                fn=mul,",
        "                args=['v2', 'p2'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'd1'],",
        '                stoichiometry={"v1": -1.0,"v2": 1.0},',
        "            )",
        "        .add_reaction(",
        '                "r2",',
        "                fn=mass_action_1s,",
        "                args=['v2', 'd2'],",
        '                stoichiometry={"v1": 1.0,"v2": -1.0},',
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_dependent_derived() -> None:
    assert generate_mxlpy_code(models.m_dependent_derived()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def constant(d1: float) -> float:",
        "    return d1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameter('p1', value=1.0)",
        "        .add_derived(",
        "                'd1',",
        "                fn=constant,",
        "                args=['p1'],",
        "            )",
        "        .add_derived(",
        "                'd2',",
        "                fn=constant,",
        "                args=['d1'],",
        "            )",
        "    )",
    ]


def test_generate_model_code_py_m_derived_stoichiometry() -> None:
    assert generate_mxlpy_code(models.m_derived_stoichiometry()).split("\n") == [
        "from mxlpy import Model, Derived, InitialAssignment",
        "",
        "def constant(v1: float) -> float:",
        "    return v1",
        "    ",
        "",
        "def r1_stoich_one_div(v1: float) -> float:",
        "    return 1.0/v1",
        "    ",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variable('v1', initial_value=1.0)",
        "        .add_reaction(",
        '                "r1",',
        "                fn=constant,",
        "                args=['v1'],",
        "                stoichiometry={\"v1\": Derived(fn=r1_stoich_one_div, args=['v1'])},",
        "            )",
        "    )",
    ]
