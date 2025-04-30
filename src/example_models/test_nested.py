from mxlpy import Model
from mxlpy.fns import constant
from mxlpy.types import Float

__all__ = ["get_model", "wrapper"]


def wrapper(x: Float) -> Float:
    return constant(x)


def get_model() -> Model:
    return (
        Model()
        .add_variables({"x": 0})
        .add_reaction("v1", wrapper, args=["x"], stoichiometry={"x": -1})
    )
