# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107, D200, D203, D400, D401

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import sympy

from modelbase2.model import Model

__all__ = ["Context", "SymbolicModel", "model_fn_to_sympy", "to_symbolic_model"]


@dataclass
class Context:
    symbols: dict[str, sympy.Symbol | sympy.Expr]
    caller: Callable


@dataclass
class SymbolicModel:
    variables: dict[str, sympy.Symbol]
    parameters: dict[str, sympy.Symbol]
    eqs: list[sympy.Expr]


def to_symbolic_model(model: Model) -> SymbolicModel:
    cache = model._create_cache()  # noqa: SLF001

    variables = dict(
        zip(model.variables, sympy.symbols(list(model.variables)), strict=True)
    )
    parameters = dict(
        zip(model.parameters, sympy.symbols(list(model.parameters)), strict=True)
    )
    symbols = variables | parameters

    for k, v in model.derived.items():
        symbols[k] = model_fn_to_sympy(v.fn, [symbols[i] for i in v.args])

    rxns = {
        k: model_fn_to_sympy(v.fn, [symbols[i] for i in v.args])
        for k, v in model.reactions.items()
    }

    eqs: dict[str, sympy.Expr] = {}
    for cpd, stoich in cache.stoich_by_cpds.items():
        for rxn, stoich_value in stoich.items():
            eqs[cpd] = (
                eqs.get(cpd, sympy.Float(0.0)) + sympy.Float(stoich_value) * rxns[rxn]  # type: ignore
            )

    for cpd, dstoich in cache.dyn_stoich_by_cpds.items():
        for rxn, der in dstoich.items():
            eqs[cpd] = eqs.get(cpd, sympy.Float(0.0)) + model_fn_to_sympy(
                der.fn,
                [symbols[i] for i in der.args] * rxns[rxn],  # type: ignore
            )  # type: ignore

    return SymbolicModel(
        variables=variables,
        parameters=parameters,
        eqs=[eqs[i] for i in cache.var_names],
    )


def model_fn_to_sympy(
    fn: Callable, model_args: list[sympy.Symbol | sympy.Expr] | None = None
) -> sympy.Expr:
    source = textwrap.dedent(inspect.getsource(fn))

    if not isinstance(fn_def := ast.parse(source).body[0], ast.FunctionDef):
        msg = "Expected a function definition"
        raise TypeError(msg)

    fn_args = [str(arg.arg) for arg in fn_def.args.args]

    sympy_expr = _handle_fn_body(
        fn_def.body,
        ctx=Context(
            symbols={name: sympy.Symbol(name) for name in fn_args},
            caller=fn,
        ),
    )

    if model_args is not None:
        sympy_expr = sympy_expr.subs(dict(zip(fn_args, model_args, strict=True)))

    return cast(sympy.Expr, sympy_expr)


def _handle_fn_body(body: list[ast.stmt], ctx: Context) -> sympy.Expr:
    pieces = []
    remaining_body = list(body)

    while remaining_body:
        node = remaining_body.pop(0)

        if isinstance(node, ast.If):
            condition = _handle_expr(node.test, ctx)
            if_expr = _handle_fn_body(node.body, ctx)
            pieces.append((if_expr, condition))

            # If there's an else clause
            if node.orelse:
                # Check if it's an elif (an If node in orelse)
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # Push the elif back to the beginning of remaining_body to process next
                    remaining_body.insert(0, node.orelse[0])
                else:
                    # It's a regular else
                    else_expr = _handle_fn_body(node.orelse, ctx)  # FIXME: copy here
                    pieces.append((else_expr, True))
                    break  # We're done with this chain

            elif not remaining_body and any(
                isinstance(n, ast.Return) for n in body[body.index(node) + 1 :]
            ):
                else_expr = _handle_fn_body(
                    body[body.index(node) + 1 :], ctx
                )  # FIXME: copy here
                pieces.append((else_expr, True))

        elif isinstance(node, ast.Return):
            if (value := node.value) is None:
                msg = "Return value cannot be None"
                raise ValueError(msg)

            expr = _handle_expr(value, ctx)
            if not pieces:
                return expr
            pieces.append((expr, True))
            break

        elif isinstance(node, ast.Assign):
            # Handle tuple assignments like c, d = a, b
            if isinstance(node.targets[0], ast.Tuple):
                # Handle tuple unpacking
                target_elements = node.targets[0].elts

                if isinstance(node.value, ast.Tuple):
                    # Direct unpacking like c, d = a, b
                    value_elements = node.value.elts
                    for target, value_expr in zip(
                        target_elements, value_elements, strict=True
                    ):
                        if isinstance(target, ast.Name):
                            ctx.symbols[target.id] = _handle_expr(value_expr, ctx)
                else:
                    # Handle potential iterable unpacking
                    value = _handle_expr(node.value, ctx)
            else:
                # Regular single assignment
                if not isinstance(target := node.targets[0], ast.Name):
                    msg = "Only single variable assignments are supported"
                    raise TypeError(msg)
                target_name = target.id
                value = _handle_expr(node.value, ctx)
                ctx.symbols[target_name] = value

    # If we have pieces to combine into a Piecewise
    if pieces:
        return sympy.Piecewise(*pieces)

    # If no return was found but we have assignments, return the last assigned variable
    for node in reversed(body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            return ctx.symbols[target_name]

    msg = "No return value found in function body"
    raise ValueError(msg)


def _handle_unaryop(node: ast.UnaryOp, ctx: Context) -> sympy.Expr:
    left = _handle_expr(node.operand, ctx)
    left = cast(Any, left)  # stupid sympy types don't allow ops on symbols

    if isinstance(node.op, ast.UAdd):
        return +left
    if isinstance(node.op, ast.USub):
        return -left

    msg = f"Operation {type(node.op).__name__} not implemented"
    raise NotImplementedError(msg)


def _handle_binop(node: ast.BinOp, ctx: Context) -> sympy.Expr:
    left = _handle_expr(node.left, ctx)
    left = cast(Any, left)  # stupid sympy types don't allow ops on symbols

    right = _handle_expr(node.right, ctx)
    right = cast(Any, right)  # stupid sympy types don't allow ops on symbols

    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Pow):
        return left**right
    if isinstance(node.op, ast.Mod):
        return left % right
    if isinstance(node.op, ast.FloorDiv):
        return left // right

    msg = f"Operation {type(node.op).__name__} not implemented"
    raise NotImplementedError(msg)


def _handle_call(node: ast.Call, ctx: Context) -> sympy.Expr:
    if not isinstance(callee := node.func, ast.Name):
        msg = "Only function calls with names are supported"
        raise TypeError(msg)

    fn_name = str(callee.id)
    parent_module = inspect.getmodule(ctx.caller)
    fns = dict(inspect.getmembers(parent_module, predicate=callable))

    return model_fn_to_sympy(
        fns[fn_name],
        model_args=[_handle_expr(i, ctx) for i in node.args],
    )


def _handle_name(node: ast.Name, ctx: Context) -> sympy.Symbol | sympy.Expr:
    return ctx.symbols[node.id]


def _handle_expr(node: ast.expr, ctx: Context) -> sympy.Expr:
    if isinstance(node, ast.UnaryOp):
        return _handle_unaryop(node, ctx)
    if isinstance(node, ast.BinOp):
        return _handle_binop(node, ctx)
    if isinstance(node, ast.Name):
        return _handle_name(node, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        # Handle chained comparisons like 1 < a < 2
        left = cast(Any, _handle_expr(node.left, ctx))
        comparisons = []

        # Build all individual comparisons from the chain
        prev_value = left
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = cast(Any, _handle_expr(comparator, ctx))

            if isinstance(op, ast.Gt):
                comparisons.append(prev_value > right)
            elif isinstance(op, ast.GtE):
                comparisons.append(prev_value >= right)
            elif isinstance(op, ast.Lt):
                comparisons.append(prev_value < right)
            elif isinstance(op, ast.LtE):
                comparisons.append(prev_value <= right)
            elif isinstance(op, ast.Eq):
                comparisons.append(prev_value == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(prev_value != right)

            prev_value = right

        # Combine all comparisons with logical AND
        result = comparisons[0]
        for comp in comparisons[1:]:
            result = sympy.And(result, comp)
        return cast(sympy.Expr, result)
    if isinstance(node, ast.Call):
        return _handle_call(node, ctx)

    # Handle conditional expressions (ternary operators)
    if isinstance(node, ast.IfExp):
        condition = _handle_expr(node.test, ctx)
        if_true = _handle_expr(node.body, ctx)
        if_false = _handle_expr(node.orelse, ctx)
        return sympy.Piecewise((if_true, condition), (if_false, True))

    msg = f"Expression type {type(node).__name__} not implemented"
    raise NotImplementedError(msg)
