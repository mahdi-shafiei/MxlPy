from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from modelbase2.typing import Axis, Figure, unwrap

if TYPE_CHECKING:
    import pandas as pd


def _fig_ax_if_neccessary(
    ax: Axis | None,
    subplot_kw: dict[str, Any] | None = None,
) -> tuple[Figure, Axis]:
    if ax is None:
        return plt.subplots(constrained_layout=True, subplot_kw=subplot_kw)

    return unwrap(ax.get_figure()), ax


def _plot_line_mean_std(
    ax: Axis,
    df: pd.DataFrame,
    color: str | None,
    label: str | None,
    alpha: float = 0.2,
) -> None:
    if color is None:
        color = f"C{len(ax.lines)}"
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ax.plot(mean, color=color, label=label)
    ax.fill_between(
        df.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )


def _plot_line_median_std(
    ax: Axis,
    cpd: pd.DataFrame,
    color: str | None,
    label: str | None,
    alpha: float = 0.2,
) -> None:
    mean = cpd.median(axis=1)
    std = cpd.std(axis=1)
    ax.plot(mean, color=color, label=label)
    ax.fill_between(
        cpd.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )
