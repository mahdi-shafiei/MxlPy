from __future__ import annotations

from modelbase2.experimental.codegen import generate_modelbase_code
from tests import models


def test_generate_modelbase_code_m_0v_1p_0d_0r() -> None:
    assert generate_modelbase_code(models.m_0v_1p_0d_0r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0})",
        "    )",
    ]


def test_generate_modelbase_code_m_0v_2p_0d_0r() -> None:
    assert generate_modelbase_code(models.m_0v_2p_0d_0r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0, 'p2': 2.0})",
        "    )",
    ]


def test_generate_modelbase_code_m_1v_0p_0d_0r() -> None:
    assert generate_modelbase_code(models.m_1v_0p_0d_0r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variables({'v1': 1.0})",
        "    )",
    ]


def test_generate_modelbase_code_m_1v_1p_1d_0r() -> None:
    assert generate_modelbase_code(models.m_1v_1p_1d_0r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def add(x: Float, y: Float) -> Float:",
        '    """Proportional function."""',
        "    return x + y",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0})",
        "        .add_variables({'v1': 1.0})",
        "        .add_derived(",
        '                "d1",',
        "                fn=add,",
        "                args=['v1', 'p1'],",
        "            )",
        "    )",
    ]


def test_generate_modelbase_code_m_1v_1p_1d_1r() -> None:
    assert generate_modelbase_code(models.m_1v_1p_1d_1r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def add(x: Float, y: Float) -> Float:",
        '    """Proportional function."""',
        "    return x + y",
        "",
        "def mass_action_1s(s1: Float, k: Float) -> Float:",
        '    """Irreversible mass action reaction with one substrate."""',
        "    return k * s1",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0})",
        "        .add_variables({'v1': 1.0})",
        "        .add_derived(",
        '                "d1",',
        "                fn=add,",
        "                args=['v1', 'p1'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'p1'],",
        '                stoichiometry={"v1": -1.0},',
        "            )",
        "    )",
    ]


def test_generate_modelbase_code_m_2v_0p_0d_0r() -> None:
    assert generate_modelbase_code(models.m_2v_0p_0d_0r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variables({'v1': 1.0, 'v2': 2.0})",
        "    )",
    ]


def test_generate_modelbase_code_m_2v_1p_1d_1r() -> None:
    assert generate_modelbase_code(models.m_2v_1p_1d_1r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def add(x: Float, y: Float) -> Float:",
        '    """Proportional function."""',
        "    return x + y",
        "",
        "def mass_action_1s(s1: Float, k: Float) -> Float:",
        '    """Irreversible mass action reaction with one substrate."""',
        "    return k * s1",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0})",
        "        .add_variables({'v1': 1.0, 'v2': 2.0})",
        "        .add_derived(",
        '                "d1",',
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


def test_generate_modelbase_code_m_2v_2p_1d_1r() -> None:
    assert generate_modelbase_code(models.m_2v_2p_1d_1r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def add(x: Float, y: Float) -> Float:",
        '    """Proportional function."""',
        "    return x + y",
        "",
        "def mass_action_1s(s1: Float, k: Float) -> Float:",
        '    """Irreversible mass action reaction with one substrate."""',
        "    return k * s1",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0, 'p2': 2.0})",
        "        .add_variables({'v1': 1.0, 'v2': 2.0})",
        "        .add_derived(",
        '                "d1",',
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


def test_generate_modelbase_code_m_2v_2p_2d_1r() -> None:
    assert generate_modelbase_code(models.m_2v_2p_2d_1r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def add(x: Float, y: Float) -> Float:",
        '    """Proportional function."""',
        "    return x + y",
        "",
        "def mul(x: Float, y: Float) -> Float:",
        '    """Multiplication function."""',
        "    return x * y",
        "",
        "def mass_action_1s(s1: Float, k: Float) -> Float:",
        '    """Irreversible mass action reaction with one substrate."""',
        "    return k * s1",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0, 'p2': 2.0})",
        "        .add_variables({'v1': 1.0, 'v2': 2.0})",
        "        .add_derived(",
        '                "d1",',
        "                fn=add,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_derived(",
        '                "d2",',
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


def test_generate_modelbase_code_m_2v_2p_2d_2r() -> None:
    assert generate_modelbase_code(models.m_2v_2p_2d_2r()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def add(x: Float, y: Float) -> Float:",
        '    """Proportional function."""',
        "    return x + y",
        "",
        "def mul(x: Float, y: Float) -> Float:",
        '    """Multiplication function."""',
        "    return x * y",
        "",
        "def mass_action_1s(s1: Float, k: Float) -> Float:",
        '    """Irreversible mass action reaction with one substrate."""',
        "    return k * s1",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0, 'p2': 2.0})",
        "        .add_variables({'v1': 1.0, 'v2': 2.0})",
        "        .add_derived(",
        '                "d1",',
        "                fn=add,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_derived(",
        '                "d2",',
        "                fn=mul,",
        "                args=['v1', 'v2'],",
        "            )",
        "        .add_reaction(",
        '                "r1",',
        "                fn=mass_action_1s,",
        "                args=['v1', 'p1'],",
        '                stoichiometry={"v1": -1.0,"v2": 1.0},',
        "            )",
        "        .add_reaction(",
        '                "r2",',
        "                fn=mass_action_1s,",
        "                args=['v2', 'p2'],",
        '                stoichiometry={"v1": 1.0,"v2": -1.0},',
        "            )",
        "    )",
    ]


def test_generate_modelbase_code_m_dependent_derived() -> None:
    assert generate_modelbase_code(models.m_dependent_derived()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def constant(x: Float) -> Float:",
        '    """Constant function."""',
        "    return x",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_parameters({'p1': 1.0})",
        "        .add_derived(",
        '                "d1",',
        "                fn=constant,",
        "                args=['p1'],",
        "            )",
        "        .add_derived(",
        '                "d2",',
        "                fn=constant,",
        "                args=['d1'],",
        "            )",
        "    )",
    ]


def test_generate_modelbase_code_m_derived_stoichiometry() -> None:
    assert generate_modelbase_code(models.m_derived_stoichiometry()).split("\n") == [
        "from modelbase2 import Model",
        "",
        "def constant(x: Float) -> Float:",
        '    """Constant function."""',
        "    return x",
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
        "        .add_variables({'v1': 1.0})",
        "        .add_reaction(",
        '                "r1",',
        "                fn=constant,",
        "                args=['v1'],",
        '                stoichiometry={"v1": Derived(name="v1", fn=constant, args=["v1"])},',
        "            )",
        "    )",
    ]
