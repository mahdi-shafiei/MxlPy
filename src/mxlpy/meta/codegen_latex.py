"""Export model to latex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy

from mxlpy.meta.source_tools import fn_to_sympy
from mxlpy.types import Derived, RateFn

__all__ = [
    "TexExport",
    "TexReaction",
    "default_init",
    "generate_latex_code",
    "get_model_tex_diff",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mxlpy import Model


cdot = r"\cdot"
empty_set = r"\emptyset"
left_right_arrows = r"\xleftrightharpoons{}"
right_arrow = r"\xrightarrow{}"
newline = r"\\" + "\n"
floatbarrier = r"\FloatBarrier"


def _list_of_symbols(args: list[str]) -> list[sympy.Symbol | sympy.Expr]:
    return [sympy.Symbol(arg) for arg in args]


def default_init[T1, T2](d: dict[T1, T2] | None) -> dict[T1, T2]:
    """Return empty dict if d is None."""
    return {} if d is None else d


def _gls(s: str) -> str:
    return rf"\gls{{{s}}}"


def _abbrev_and_full(s: str) -> str:
    return rf"\acrfull{{{s}}}"


def _gls_short(s: str) -> str:
    return rf"\acrshort{{{s}}}"


def _gls_full(s: str) -> str:
    return rf"\acrlong{{{s}}}"


def _rename_latex(s: str) -> str:
    if s[0].isdigit():
        s = s[1:]
        if s[0] == "-":
            s = s[1:]
    return (
        s.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("*", "")
    )


def _escape_non_math(s: str) -> str:
    return s.replace("_", r"\_")


def _fn_to_latex(fn: Callable, arg_names: list[str]) -> str:
    return sympy.latex(fn_to_sympy(fn, _list_of_symbols(arg_names)))


def _table(
    headers: list[str],
    rows: list[list[str]],
    n_columns: int,
    label: str,
    short_desc: str,
    long_desc: str,
) -> str:
    columns = "|".join(["c"] * n_columns)
    tab = "    "

    return "\n".join(
        [
            r"\begin{longtable}" + f"{{{columns}}}",
            tab + " & ".join(headers) + r" \\",
            tab + r"\hline",
            tab + r"\endhead",
        ]
        + [tab + " & ".join(i) + r" \\" for i in rows]
        + [
            tab + rf"\caption[{short_desc}]{{{long_desc}}}",
            tab + rf"\label{{table:{label}}}",
            r"\end{longtable}",
        ]
    )


def _label(content: str) -> str:
    return rf"\label{{{content}}}"


def _dmath(content: str) -> str:
    return rf"""\begin{{dmath*}}
    {content}
\end{{dmath*}}"""


# def _dmath_il(content: str) -> str:
#     return rf"\begin{{dmath*}}{content}\end{{dmath*}}"


def _part(s: str) -> str:
    # depth = -1
    return floatbarrier + rf"\part{{{s}}}"


def _chapter(s: str) -> str:
    # depth = 0
    return floatbarrier + rf"\part{{{s}}}"


def _section(s: str) -> str:
    # depth = 1
    return floatbarrier + rf"\section{{{s}}}"


def _section_(s: str) -> str:
    # depth = 1
    return floatbarrier + rf"\section*{{{s}}}"


def _subsection(s: str) -> str:
    # depth = 2
    return floatbarrier + rf"\subsection{{{s}}}"


def _subsection_(s: str) -> str:
    # depth = 2
    return floatbarrier + rf"\subsection*{{{s}}}"


def _subsubsection(s: str) -> str:
    # depth = 3
    return floatbarrier + rf"\subsubsection{{{s}}}"


def _subsubsection_(s: str) -> str:
    # depth = 3
    return floatbarrier + rf"\subsubsection*{{{s}}}"


def _paragraph(s: str) -> str:
    # depth = 4
    return rf"\paragraph{{{s}}}"


def _subparagraph(s: str) -> str:
    # depth = 5
    return rf"\subparagraph{{{s}}}"


def _math_il(s: str) -> str:
    return f"${s}$"


def _math(s: str) -> str:
    return f"$${s}$$"


def _mathrm(s: str) -> str:
    return rf"\mathrm{{{s}}}"


def _bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def _clearpage() -> str:
    return r"\clearpage"


def _latex_list(rows: list[str]) -> str:
    return "\n\n".join(rows)


def _latex_list_as_sections(
    rows: list[tuple[str, str]], sec_fn: Callable[[str], str]
) -> str:
    return "\n\n".join(
        [
            "\n".join(
                (
                    sec_fn(_escape_non_math(name)),
                    content,
                )
            )
            for name, content in rows
        ]
    )


def _latex_list_as_bold(rows: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        [
            "\n".join(
                (
                    _bold(_escape_non_math(name)) + r"\\",
                    content,
                    r"\vspace{20pt}",
                )
            )
            for name, content in rows
        ]
    )


def _stoichiometries_to_latex(stoich: Mapping[str, float | Derived]) -> str:
    def optional_factor(k: str, v: float) -> str:
        if v == 1:
            return _mathrm(k)
        if v == -1:
            return f"-{_mathrm(k)}"
        return f"{v} {cdot} {_mathrm(k)}"

    def latex_for_empty(s: str) -> str:
        if len(s) == 0:
            return empty_set
        return s

    line = []
    for k, v in stoich.items():
        if isinstance(v, int | float):
            line.append(optional_factor(k, v))
        else:
            line.append(_fn_to_latex(v.fn, [_rename_latex(i) for i in v.args]))

    line_str = f"{line[0]}"
    for element in line[1:]:
        if element.startswith("-"):
            line_str += f" {element}"
        else:
            line_str += f" + {element}"
    return _math_il(line_str)


@dataclass
class TexReaction:
    """Collection for reaction."""

    fn: RateFn
    args: list[str]


@dataclass
class TexExport:
    """Container for LaTeX export."""

    parameters: dict[str, float]
    variables: dict[str, float]
    derived: dict[str, Derived]
    reactions: dict[str, TexReaction]
    stoichiometries: dict[str, Mapping[str, float | Derived]]

    @staticmethod
    def _diff_parameters(
        p1: dict[str, float],
        p2: dict[str, float],
    ) -> dict[str, float]:
        return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

    @staticmethod
    def _diff_variables(p1: dict[str, float], p2: dict[str, float]) -> dict[str, float]:
        return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

    @staticmethod
    def _diff_derived(
        p1: dict[str, Derived],
        p2: dict[str, Derived],
    ) -> dict[str, Derived]:
        return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

    @staticmethod
    def _diff_reactions(
        p1: dict[str, TexReaction],
        p2: dict[str, TexReaction],
    ) -> dict[str, TexReaction]:
        return {k: v for k, v in p2.items() if k not in p1 or p1[k] != v}

    def __sub__(self, other: object) -> TexExport:
        """Return difference of two tex exports."""
        if not isinstance(other, TexExport):
            raise TypeError

        return TexExport(
            parameters=self._diff_parameters(self.parameters, other.parameters),
            variables=self._diff_variables(self.variables, other.variables),
            derived=self._diff_derived(self.derived, other.derived),
            reactions=self._diff_reactions(self.reactions, other.reactions),
            stoichiometries={
                k: v
                for k, v in other.stoichiometries.items()
                if self.stoichiometries.get(k, {}) != v
            },
        )

    def rename_with_glossary(self, gls: dict[str, str]) -> TexExport:
        """Rename all elements according to glossary."""

        def _add_gls_if_found(k: str) -> str:
            if (new := gls.get(k)) is not None:
                return _abbrev_and_full(new)
            return k

        return TexExport(
            parameters={gls.get(k, k): v for k, v in self.parameters.items()},
            variables={gls.get(k, k): v for k, v in self.variables.items()},
            derived={
                gls.get(k, k): Derived(
                    name=k, fn=v.fn, args=[gls.get(i, i) for i in v.args]
                )
                for k, v in self.derived.items()
            },
            reactions={
                _add_gls_if_found(k): TexReaction(
                    fn=v.fn,
                    args=[gls.get(i, i) for i in v.args],
                )
                for k, v in self.reactions.items()
            },
            stoichiometries={
                _add_gls_if_found(k): {gls.get(k2, k2): v2 for k2, v2 in v.items()}
                for k, v in self.stoichiometries.items()
            },
        )

    def export_variables(self) -> str:
        """Export variables."""
        return _table(
            headers=["Model name", "Initial concentration"],
            rows=[
                [
                    k,
                    f"{v:.2e}",
                ]
                for k, v in self.variables.items()
            ],
            n_columns=2,
            label="model-vars",
            short_desc="Model variables",
            long_desc="Model variables",
        )

    def export_parameters(self) -> str:
        """Export parameters."""
        return _table(
            headers=["Parameter name", "Parameter value"],
            rows=[
                [_math_il(_mathrm(_escape_non_math(_rename_latex(k)))), f"{v:.2e}"]
                for k, v in sorted(self.parameters.items())
            ],
            n_columns=2,
            label="model-pars",
            short_desc="Model parameters",
            long_desc="Model parameters",
        )

    def export_derived(self) -> str:
        """Export derived quantities."""
        return _latex_list(
            rows=[
                _dmath(
                    f"{_rename_latex(k)} = {_fn_to_latex(v.fn, [_rename_latex(i) for i in v.args])}"
                )
                for k, v in sorted(self.derived.items())
            ]
        )

    def export_reactions(self) -> str:
        """Export reactions."""
        return _latex_list(
            rows=[
                _dmath(
                    f"{_rename_latex(k)} = {_fn_to_latex(v.fn, [_rename_latex(i) for i in v.args])}"
                )
                for k, v in sorted(self.reactions.items())
            ]
        )

    def export_stoichiometries(self) -> str:
        """Export stoichiometries."""
        return _table(
            headers=["Rate name", "Stoichiometry"],
            rows=[
                [
                    _escape_non_math(_rename_latex(k)),
                    _stoichiometries_to_latex(v),
                ]
                for k, v in sorted(self.stoichiometries.items())
            ],
            n_columns=2,
            label="model-stoichs",
            short_desc="Model stoichiometries.",
            long_desc="Model stoichiometries.",
        )

    def export_all(self) -> str:
        """Export all model parts."""
        sections = []
        if len(self.variables) > 0:
            sections.append(
                (
                    "Variables",
                    self.export_variables(),
                )
            )
        if len(self.parameters) > 0:
            sections.append(
                (
                    "Parameters",
                    self.export_parameters(),
                )
            )
        if len(self.derived) > 0:
            sections.append(
                (
                    "Derived",
                    self.export_derived(),
                )
            )
        if len(self.reactions) > 0:
            sections.append(
                (
                    "Reactions",
                    self.export_reactions(),
                )
            )
            sections.append(
                (
                    "Stoichiometries",
                    self.export_stoichiometries(),
                )
            )
        return _latex_list_as_sections(sections, _subsubsection_)

    def export_document(
        self,
        author: str = "mxlpy",
        title: str = "Model construction",
    ) -> str:
        """Export latex document."""
        content = self.export_all()
        return rf"""\documentclass{{article}}
\usepackage[english]{{babel}}
\usepackage[a4paper,top=2cm,bottom=2cm,left=2cm,right=2cm,marginparwidth=1.75cm]{{geometry}}
\usepackage{{amsmath, amssymb, array, booktabs,
            breqn, caption, longtable, mathtools, placeins,
            ragged2e, tabularx, titlesec, titling}}
\newcommand{{\sectionbreak}}{{\clearpage}}
\setlength{{\parindent}}{{0pt}}

\title{{{title}}}
\date{{}} % clear date
\author{{{author}}}
\begin{{document}}
\maketitle
{content}
\end{{document}}
"""


def _to_tex_export(self: Model) -> TexExport:
    return TexExport(
        parameters=self.parameters,
        variables=self.variables,
        derived=self.derived,
        reactions={k: TexReaction(v.fn, v.args) for k, v in self.reactions.items()},
        stoichiometries={k: v.stoichiometry for k, v in self.reactions.items()},
    )


def generate_latex_code(
    model: Model,
    gls: dict[str, str] | None = None,
) -> str:
    """Export as LaTeX."""
    gls = default_init(gls)
    return _to_tex_export(model).rename_with_glossary(gls).export_document()


def get_model_tex_diff(
    m1: Model,
    m2: Model,
    gls: dict[str, str] | None = None,
) -> str:
    """Create LaTeX diff of two models."""
    gls = default_init(gls)
    section_label = "sec:model-diff"

    return f"""{" start autogenerated ":%^60}
{_clearpage()}
{_subsubsection("Model changes")}{_label(section_label)}
{((_to_tex_export(m1) - _to_tex_export(m2)).rename_with_glossary(gls).export_all())}
{_clearpage()}
{" end autogenerated ":%^60}
"""
