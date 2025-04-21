"""Module to export models as code."""

import ast
import inspect
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator

from modelbase2.model import Model
from modelbase2.types import Derived

__all__ = [
    "DocstringRemover",
    "IdentifierReplacer",
    "ReturnRemover",
    "conditional_join",
    "generate_model_code_py",
    "generate_modelbase_code",
    "get_fn_source",
    "handle_fn",
]


class IdentifierReplacer(ast.NodeTransformer):
    """Replace identifiers in an AST."""

    def __init__(self, mapping: dict[str, str]) -> None:
        """Initialize the transformer with a mapping."""
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        """Replace the identifier with the mapped value."""
        return ast.Name(
            id=self.mapping.get(node.id, node.id),
            ctx=node.ctx,
        )


class DocstringRemover(ast.NodeTransformer):
    """Remove docstrings from an AST."""

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:  # noqa: N802
        """Remove docstrings."""
        if isinstance(const := node.value, ast.Constant) and isinstance(
            const.value, str
        ):
            return None
        return node


class ReturnRemover(ast.NodeTransformer):
    """Remove return statements from an AST."""

    def visit_Return(self, node: ast.Return) -> ast.expr | None:  # noqa: N802
        """Remove return statements."""
        return node.value


def get_fn_source(fn: Callable) -> ast.FunctionDef:
    """Get the source code of a function as an AST."""
    import inspect
    import textwrap

    import dill

    try:
        source = inspect.getsource(fn)
    except OSError:  # could not get source code
        source = dill.source.getsource(fn)

    tree = ast.parse(textwrap.dedent(source))
    if not isinstance(fn_def := tree.body[0], ast.FunctionDef):
        msg = "Not a function"
        raise TypeError(msg)
    return fn_def


def handle_fn(fn: Callable, args: list[str]) -> str:
    """Get the source code of a function, removing docstrings and return statements."""
    tree = get_fn_source(fn)

    argmap = dict(zip([i.arg for i in tree.args.args], args, strict=True))
    tree = DocstringRemover().visit(tree)
    tree = IdentifierReplacer(argmap).visit(tree)
    tree = ReturnRemover().visit(tree)
    return ast.unparse(tree.body)


def conditional_join[T](
    iterable: Iterable[T],
    question: Callable[[T], bool],
    true_pat: str,
    false_pat: str,
) -> str:
    """Join an iterable, applying a pattern to each element based on a condition."""

    def inner(it: Iterator[T]) -> Generator[str, None, None]:
        yield str(next(it))
        while True:
            try:
                el = next(it)
                if question(el):
                    yield f"{true_pat}{el}"
                else:
                    yield f"{false_pat}{el}"
            except StopIteration:
                break

    return "".join(inner(iter(iterable)))


def generate_modelbase_code(model: Model) -> str:
    """Generate a modelbase model from a model."""
    functions = {}

    # Variables and parameters
    variables = model.variables
    parameters = model.parameters

    # Derived
    derived_source = []
    for k, v in model.derived.items():
        fn = v.fn
        fn_name = fn.__name__
        functions[fn_name] = inspect.getsource(fn)

        derived_source.append(
            f""".add_derived(
                "{k}",
                fn={fn_name},
                args={v.args},
            )"""
        )

    # Reactions
    reactions_source = []
    for k, v in model.reactions.items():
        fn = v.fn
        fn_name = fn.__name__
        functions[fn_name] = inspect.getsource(fn)
        reactions_source.append(
            f"""        .add_reaction(
                "{k}",
                fn={fn_name},
                args={v.args},
                stoichiometry={v.stoichiometry},
            )"""
        )

    # Surrogates
    if len(model._surrogates) > 0:  # noqa: SLF001
        warnings.warn(
            "Generating code for Surrogates not yet supported.",
            stacklevel=1,
        )

    # Combine all the sources
    functions_source = "\n".join(functions.values())
    source = [
        "from modelbase2 import Model\n",
        functions_source,
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
    ]
    if len(parameters) > 0:
        source.append(f"        .add_parameters({parameters})")
    if len(variables) > 0:
        source.append(f"        .add_variables({variables})")
    if len(derived_source) > 0:
        source.append("\n".join(derived_source))
    if len(reactions_source) > 0:
        source.append("\n".join(reactions_source))

    source.append("    )")

    return "\n".join(source)


def generate_model_code_py(model: Model) -> str:
    """Transform the model into a single function, inlining the function calls."""
    # Variables
    variables = model.variables
    variable_source = "    {} = y".format(", ".join(variables))

    # Parameters
    parameter_source = "\n".join(f"    {k} = {v}" for k, v in model.parameters.items())

    # Derived
    derived_source = []
    for name, derived in model.derived.items():
        derived_source.append(f"    {name} = {handle_fn(derived.fn, derived.args)}")

    # Reactions
    reactions = []
    for name, rxn in model.reactions.items():
        reactions.append(f"    {name} = {handle_fn(rxn.fn, rxn.args)}")

    # Stoichiometries
    stoichiometries = {}
    for rxn_name, rxn in model.reactions.items():
        for cpd_name, factor in rxn.stoichiometry.items():
            if isinstance(factor, Derived):
                src = f"{handle_fn(factor.fn, factor.args)} * {rxn_name}"
            elif factor == 1:
                src = rxn_name
            elif factor == -1:
                src = f"- {rxn_name}"
            else:
                src = f"{factor} * {rxn_name}"
            stoichiometries.setdefault(cpd_name, []).append(src)
    stoich_source = []
    for variable, stoich in stoichiometries.items():
        stoich_source.append(
            f"    d{variable}dt = {conditional_join(stoich, lambda x: x.startswith('-'), ' ', ' + ')}"
        )

    # Surrogates
    if len(model._surrogates) > 0:  # noqa: SLF001
        warnings.warn(
            "Generating code for Surrogates not yet supported.",
            stacklevel=1,
        )

    # Combine all the sources
    source = [
        "from collections.abc import Iterable\n",
        "from modelbase2.types import Float\n",
        "def model(t: Float, y: Float) -> Iterable[Float]:",
        variable_source,
        parameter_source,
        *derived_source,
        *reactions,
        *stoich_source,
        "    return {}".format(
            ", ".join(f"d{i}dt" for i in variables),
        ),
    ]

    return "\n".join(source)
