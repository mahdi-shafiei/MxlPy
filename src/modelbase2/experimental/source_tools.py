"""Tools for working with python source files."""

import ast
import inspect
import textwrap
from collections.abc import Callable

import dill

__all__ = [
    "get_fn_ast",
    "get_fn_source",
]


def get_fn_source(fn: Callable) -> str:
    """Get the string representation of a function."""
    try:
        return inspect.getsource(fn)
    except OSError:  # could not get source code
        return dill.source.getsource(fn)


def get_fn_ast(fn: Callable) -> ast.FunctionDef:
    """Get the source code of a function as an AST."""
    tree = ast.parse(textwrap.dedent(get_fn_source(fn)))
    if not isinstance(fn_def := tree.body[0], ast.FunctionDef):
        msg = "Not a function"
        raise TypeError(msg)
    return fn_def
