"""Export model to latex.

This isn't implemented yet, as latexify doesn't yet support Python 3.12.
Either write the parser / ast transform myself or wait for the library to be updated.
"""

import ast
import inspect
from typing import Callable

__all__ = ["parse_fn"]


def parse_fn(fn: Callable) -> list[ast.stmt]:
    return ast.parse(inspect.getsource(fn)).body


# from __future__ import annotations

# import ast
# from dataclasses import dataclass
# from typing import Callable, TypeVar, cast

# import latexify
# from modelbase.core.utils import get_function_source_code
# from modelbase.ode import Model

# from modelbase2 import Model

# T1 = TypeVar("T1")
# T2 = TypeVar("T2")


# cdot = r"\cdot"
# empty_set = r"\emptyset"
# left_right_arrows = r"\xleftrightharpoons{}"
# right_arrow = r"\xrightarrow{}"
# newline = r"\\" + "\n"
# floatbarrier = r"\FloatBarrier"


# def default_init(d: dict[T1, T2] | None) -> dict[T1, T2]:
#     return {} if d is None else d


# def _gls(s: str) -> str:
#     return rf"\gls{{{s}}}"


# def _abbrev_and_full(s: str) -> str:
#     return rf"\acrfull{{{s}}}"


# def _gls_short(s: str) -> str:
#     return rf"\acrshort{{{s}}}"


# def _gls_full(s: str) -> str:
#     return rf"\acrlong{{{s}}}"


# def _rename_latex(s: str) -> str:
#     if s[0].isdigit():
#         s = s[1:]
#         if s[0] == "-":
#             s = s[1:]
#     return (
#         s.replace(" ", "_")
#         .replace("(", "")
#         .replace(")", "")
#         .replace("-", "_")
#         .replace("*", "")
#     )


# def _escape_non_math(s: str) -> str:
#     return s.replace("_", r"\_")


# def _fn_to_latex(fn: Callable, arg_names: list[str]) -> str:
#     code = cast(str, get_function_source_code(function=fn))
#     src = cast(ast.Module, ast.parse(code))
#     fn_def = cast(ast.FunctionDef, src.body[0])
#     args: list[str] = [i.arg for i in fn_def.args.args]
#     arg_mapping: dict[str, str] = dict(zip(args, arg_names))
#     return cast(
#         str,
#         latexify.expression(
#             fn,
#             identifiers=arg_mapping,
#             reduce_assignments=True,
#         )._latex,
#     )


# def _table(
#     headers: list[str],
#     rows: list[list[str]],
#     n_columns: int,
#     label: str,
#     short_desc: str,
#     long_desc: str,
# ) -> str:
#     columns = "|".join(["c"] * n_columns)
#     tab = "    "

#     return "\n".join(
#         [
#             r"\begin{longtable}" + f"{{{columns}}}",
#             tab + " & ".join(headers) + r" \\",
#             tab + r"\hline",
#             tab + r"\endhead",
#         ]
#         + [tab + " & ".join(i) + r" \\" for i in rows]
#         + [
#             tab + rf"\caption[{short_desc}]{{{long_desc}}}",
#             tab + rf"\label{{table:{label}}}",
#             r"\end{longtable}",
#         ]
#     )


# def _label(content: str) -> str:
#     return rf"\label{{{content}}}"


# def _dmath(content: str) -> str:
#     return rf"""\begin{{dmath*}}
#     {content}
# \end{{dmath*}}"""


# # def _dmath_il(content: str) -> str:
# #     return rf"\begin{{dmath*}}{content}\end{{dmath*}}"


# def _part(s: str) -> str:
#     # depth = -1
#     return floatbarrier + rf"\part{{{s}}}"


# def _chapter(s: str) -> str:
#     # depth = 0
#     return floatbarrier + rf"\part{{{s}}}"


# def _section(s: str) -> str:
#     # depth = 1
#     return floatbarrier + rf"\section{{{s}}}"


# def _section_(s: str) -> str:
#     # depth = 1
#     return floatbarrier + rf"\section*{{{s}}}"


# def _subsection(s: str) -> str:
#     # depth = 2
#     return floatbarrier + rf"\subsection{{{s}}}"


# def _subsection_(s: str) -> str:
#     # depth = 2
#     return floatbarrier + rf"\subsection*{{{s}}}"


# def _subsubsection(s: str) -> str:
#     # depth = 3
#     return floatbarrier + rf"\subsubsection{{{s}}}"


# def _subsubsection_(s: str) -> str:
#     # depth = 3
#     return floatbarrier + rf"\subsubsection*{{{s}}}"


# def _paragraph(s: str) -> str:
#     # depth = 4
#     return rf"\paragraph{{{s}}}"


# def _subparagraph(s: str) -> str:
#     # depth = 5
#     return rf"\subparagraph{{{s}}}"


# def _math_il(s: str) -> str:
#     return f"${s}$"


# def _math(s: str) -> str:
#     return f"$${s}$$"


# def _mathrm(s: str) -> str:
#     return rf"\mathrm{{{s}}}"


# def _bold(s: str) -> str:
#     return rf"\textbf{{{s}}}"


# def _clearpage() -> str:
#     return r"\clearpage"


# def _latex_sectioned_list(
#     rows: list[tuple[str, str]], sec_fn: Callable[[str], str]
# ) -> str:
#     return "\n\n".join(
#         [
#             "\n".join(
#                 (
#                     sec_fn(_escape_non_math(name)),
#                     content,
#                 )
#             )
#             for name, content in rows
#         ]
#     )


# def _latex_list_as_bold(rows: list[tuple[str, str]]) -> str:
#     return "\n\n".join(
#         [
#             "\n".join(
#                 (
#                     _bold(_escape_non_math(name)) + r"\\",
#                     content,
#                     r"\vspace{20pt}",
#                 )
#             )
#             for name, content in rows
#         ]
#     )


# def _stoichiometries_to_latex(stoich: dict[str, float], reversible: bool) -> str:
#     def optional_factor(k: str, v: float) -> str:
#         if v != 1:
#             return f"{v} {cdot} {_mathrm(k)}"
#         return _mathrm(k)

#     def latex_for_empty(s: str) -> str:
#         if len(s) == 0:
#             return empty_set
#         return s

#     substrates: dict[str, float] = {}
#     products: dict[str, float] = {}
#     for k, v in stoich.items():
#         if v < 0:
#             substrates[k] = -v
#         else:
#             products[k] = v

#     arrow = left_right_arrows if reversible else right_arrow
#     s = " + ".join(optional_factor(k, v) for k, v in substrates.items())
#     p = " + ".join(optional_factor(k, v) for k, v in products.items())
#     s = latex_for_empty(s)
#     p = latex_for_empty(p)

#     return f"${s} {arrow} {p}$"


# @dataclass
# class TexDerivedParameter:
#     fn: Callable
#     args: list[str]


# @dataclass
# class TexDerivedVariable:
#     fn: Callable
#     args: list[str]


# @dataclass
# class TexReaction:
#     fn: Callable
#     args: list[str]


# @dataclass
# class TexExport:
#     parameters: dict[str, float]
#     derived_parameters: dict[str, TexDerivedParameter]
#     variables: list[str]
#     derived_variables: dict[str, TexDerivedVariable]
#     reactions: dict[str, TexReaction]
#     stoichiometries: dict[str, dict[str, float]]

#     @staticmethod
#     def _diff_parameters(
#         p1: dict[str, float],
#         p2: dict[str, float],
#     ) -> dict[str, float]:
#         return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

#     @staticmethod
#     def _diff_derived_parameters(
#         p1: dict[str, TexDerivedParameter],
#         p2: dict[str, TexDerivedParameter],
#     ) -> dict[str, TexDerivedParameter]:
#         return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

#     @staticmethod
#     def _diff_variables(p1: list[str], p2: list[str]) -> list[str]:
#         p1s = set(p1)
#         return [k for k in p2 if k not in p1s]

#     @staticmethod
#     def _diff_derived_variables(
#         p1: dict[str, TexDerivedVariable],
#         p2: dict[str, TexDerivedVariable],
#     ) -> dict[str, TexDerivedVariable]:
#         return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

#     @staticmethod
#     def _diff_reactions(
#         p1: dict[str, TexReaction],
#         p2: dict[str, TexReaction],
#     ) -> dict[str, TexReaction]:
#         return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

#     def __sub__(self, other: object) -> "TexExport":
#         if not isinstance(other, TexExport):
#             raise ValueError

#         return TexExport(
#             parameters=self._diff_parameters(self.parameters, other.parameters),
#             derived_parameters=self._diff_derived_parameters(
#                 self.derived_parameters, other.derived_parameters
#             ),
#             variables=self._diff_variables(self.variables, other.variables),
#             derived_variables=self._diff_derived_variables(
#                 self.derived_variables, other.derived_variables
#             ),
#             reactions=self._diff_reactions(self.reactions, other.reactions),
#             stoichiometries={
#                 k: v
#                 for k, v in other.stoichiometries.items()
#                 if self.stoichiometries.get(k, {}) != v
#             },
#         )

#     def rename_with_glossary(self, gls: dict[str, str]) -> "TexExport":
#         def _add_gls_if_found(k: str) -> str:
#             if (new := gls.get(k, None)) is not None:
#                 return _abbrev_and_full(new)
#             return k

#         return TexExport(
#             variables=[gls.get(k, k) for k in self.variables],
#             parameters={gls.get(k, k): v for k, v in self.parameters.items()},
#             derived_parameters={
#                 gls.get(k, k): TexDerivedParameter(
#                     fn=v.fn, args=[gls.get(i, i) for i in v.args]
#                 )
#                 for k, v in self.derived_parameters.items()
#             },
#             derived_variables={
#                 gls.get(k, k): TexDerivedVariable(
#                     fn=v.fn, args=[gls.get(i, i) for i in v.args]
#                 )
#                 for k, v in self.derived_variables.items()
#             },
#             reactions={
#                 _add_gls_if_found(k): TexReaction(
#                     fn=v.fn,
#                     args=[gls.get(i, i) for i in v.args],
#                 )
#                 for k, v in self.reactions.items()
#             },
#             stoichiometries={
#                 _add_gls_if_found(k): {gls.get(k2, k2): v2 for k2, v2 in v.items()}
#                 for k, v in self.stoichiometries.items()
#             },
#         )

#     def export_variables(self, *, key: str, desc: str) -> str:
#         return _table(
#             headers=["Model name", "Name"],
#             rows=[
#                 [
#                     k,
#                     rf"\acrlong{{{k}}}",
#                 ]
#                 for k in sorted(self.variables)
#             ],
#             n_columns=2,
#             label=f"vars-{key}",
#             short_desc=desc,
#             long_desc=desc,
#         )

#     def export_parameters(self, *, key: str, desc: str) -> str:
#         return _table(
#             headers=["Parameter name", "Parameter value"],
#             rows=[
#                 [_math_il(_mathrm(_escape_non_math(_rename_latex(k)))), f"{v:.2e}"]
#                 for k, v in sorted(self.parameters.items())
#             ],
#             n_columns=2,
#             label=f"pars-{key}",
#             short_desc=desc,
#             long_desc=desc,
#         )

#     def export_derived_parameters(self, *, key: str, desc: str) -> str:
#         return _table(
#             headers=["Name", "Equation"],
#             rows=[
#                 [
#                     _rename_latex(k),
#                     _math_il(_fn_to_latex(v.fn, [_rename_latex(i) for i in v.args])),
#                 ]
#                 for k, v in sorted(self.derived_parameters.items())
#             ],
#             n_columns=2,
#             label=f"dpars-{key}",
#             short_desc=desc,
#             long_desc=desc,
#         )

#     def export_derived_variables(self, *, key: str, desc: str) -> str:
#         return _latex_list_as_bold(
#             rows=[
#                 (
#                     _rename_latex(k),
#                     _dmath(_fn_to_latex(v.fn, [_rename_latex(i) for i in v.args])),
#                 )
#                 for k, v in sorted(self.derived_variables.items())
#             ]
#         )

#     def export_reactions(self, *, key: str, desc: str) -> str:
#         return _latex_list_as_bold(
#             rows=[
#                 (
#                     k,
#                     _dmath(_fn_to_latex(v.fn, [_rename_latex(i) for i in v.args])),
#                 )
#                 for k, v in sorted(self.reactions.items())
#             ]
#         )

#     def export_stoichiometries(self, *, key: str, desc: str) -> str:
#         return _table(
#             headers=["Rate name", "Stoichiometry"],
#             rows=[
#                 [
#                     k,
#                     _stoichiometries_to_latex(v, reversible=True),
#                 ]
#                 for k, v in sorted(self.stoichiometries.items())
#             ],
#             n_columns=2,
#             label=f"stoichs-{key}",
#             short_desc=desc,
#             long_desc=desc,
#         )

#     def export_all(self, key: str, name: str = "") -> str:
#         sections = []
#         if len(self.variables) > 0:
#             sections.append(
#                 (
#                     "Variables",
#                     self.export_variables(key=key, desc=f"Variables of {name}"),
#                 )
#             )
#         if len(self.parameters) > 0:
#             sections.append(
#                 (
#                     "Parameters",
#                     self.export_parameters(key=key, desc=f"Parameters of {name}"),
#                 )
#             )
#         if len(self.derived_parameters) > 0:
#             sections.append(
#                 (
#                     "Derived Parameters",
#                     self.export_derived_parameters(
#                         key=key, desc=f"Derived parameters of {name}"
#                     ),
#                 )
#             )
#         if len(self.derived_variables) > 0:
#             sections.append(
#                 (
#                     "Algebraic Modules",
#                     self.export_derived_variables(
#                         key=key, desc=f"Algebraic modules of {name}"
#                     ),
#                 )
#             )
#         if len(self.reactions) > 0:
#             sections.append(
#                 (
#                     "Reactions",
#                     self.export_reactions(key=key, desc=f"Reactions of {name}"),
#                 )
#             )
#             sections.append(
#                 (
#                     "Stoichiometries",
#                     self.export_stoichiometries(
#                         key=key, desc=f"Stoichiometries of {name}"
#                     ),
#                 )
#             )
#         return _latex_sectioned_list(sections, _subsubsection_)

#     def export_document(
#         self,
#         author: str,
#         title: str = "Model construction",
#     ) -> str:
#         content = self.export_all(key="model")
#         return rf"""\documentclass{{article}}
# \usepackage[english]{{babel}}
# \usepackage[a4paper,top=2cm,bottom=2cm,left=2cm,right=2cm,marginparwidth=1.75cm]{{geometry}}
# \usepackage{{amsmath, amssymb, array, booktabs,
#             breqn, caption, longtable, mathtools,
#             ragged2e, tabularx, titlesec, titling}}
# \newcommand{{\sectionbreak}}{{\clearpage}}
# \setlength{{\parindent}}{{0pt}}

# \title{{{title}}}
# \date{{}} % clear date
# \author{{{author}}}
# \begin{{document}}
# \maketitle
# \tableofcontents
# {content}
# \end{{document}}
# """


# def to_tex_export(self: Model) -> TexExport:
#     derived_parameters = {
#         k: TexDerivedParameter(v.function, v.parameters)
#         for k, v in self.derived_parameters.items()
#     }
#     derived_variables = {
#         k: TexDerivedVariable(v.function, v.args)
#         for k, v in self.algebraic_modules.items()
#     }
#     reactions = {
#         k: TexReaction(
#             fn=v.function,
#             args=v.args,
#         )
#         for k, v in self.rates.items()
#     }
#     stoichiometries = {k: v for k, v in self.stoichiometries.items()}

#     return TexExport(
#         parameters=self.parameters,
#         derived_parameters=derived_parameters,
#         variables=self.compounds,
#         derived_variables=derived_variables,
#         reactions=reactions,
#         stoichiometries=stoichiometries,
#     )


# def print_model_tex(
#     model: Model, gls: dict[str, str] | None = None, key: str = "", name: str = ""
# ) -> None:
#     gls = default_init(gls)
#     print(to_tex_export(model).rename_with_glossary(gls).export_all(key=key, name=name))


# def get_model_tex_diff(
#     m1: Model,
#     m2: Model,
#     gls: dict[str, str] | None = None,
#     *,
#     key: str = "",
#     name: str = "",
# ) -> str:
#     gls = default_init(gls)
#     section_label = f"sec:{key}"

#     return f"""{' start autogenerated ':%^60}
# {_clearpage()}
# {_subsubsection(f'{name} changes')}{_label(section_label)}
# {(
#     (to_tex_export(m1) - to_tex_export(m2))
#     .rename_with_glossary(gls)
#     .export_all(key=key, name=name)
# )}
# {_clearpage()}
# {' end autogenerated ':%^60}
# """
