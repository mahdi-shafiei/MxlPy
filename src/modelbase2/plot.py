import itertools as it
import math
from typing import cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import (
    LogNorm,
    Normalize,
    SymLogNorm,
    colorConverter,  # type: ignore
)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from modelbase2.label_map import LabelMapper
from modelbase2.linear_label_map import LinearLabelMapper
from modelbase2.types import Array, default_if_none

type FigAx = tuple[Figure, Axes]
type FigAxs = tuple[Figure, list[Axes]]


##########################################################################
# Helpers
##########################################################################


def _relative_luminance(color: Array) -> float:
    """Calculate the relative luminance of a color."""
    rgb = colorConverter.to_rgba_array(color)[:, :3]

    # If RsRGB <= 0.03928 then R = RsRGB/12.92 else R = ((RsRGB+0.055)/1.055) ^ 2.4
    rsrgb = np.where(
        rgb <= 0.03928,  # noqa: PLR2004
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    rel_luminance: Array = np.matmul(rsrgb, [0.2126, 0.7152, 0.0722])
    return rel_luminance[0]


def _get_norm(vmin: float, vmax: float) -> Normalize:
    if vmax < 1000 and vmin > -1000:  # noqa: PLR2004
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif vmin <= 0:
        norm = SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    return norm


def _norm_with_zero_center(df: pd.DataFrame) -> Normalize:
    v = max(abs(df.min().min()), abs(df.max().max()))
    return _get_norm(vmin=-v, vmax=v)


def _partition_by_order_of_magnitude(s: pd.Series) -> list[list[str]]:
    return [
        i.to_list()
        for i in np.floor(np.log10(s)).to_frame(name=0).groupby(0)[0].groups.values()  # type: ignore
    ]


def _split_large_groups[T](groups: list[list[T]], max_size: int) -> list[list[T]]:
    return list(
        it.chain(
            *(
                (
                    [group]
                    if len(group) < max_size
                    else [  # type: ignore
                        list(i)
                        for i in np.array_split(group, math.ceil(len(group) / max_size))  # type: ignore
                    ]
                )
                for group in groups
            )
        )
    )  # type: ignore


def _default_color(ax: Axes, color: str | None) -> str:
    return f"C{len(ax.lines)}" if color is None else color


def _default_labels(
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
) -> None:
    ax.set_xlabel("Add a label / unit" if xlabel is None else xlabel)
    ax.set_ylabel("Add a label / unit" if ylabel is None else ylabel)
    if isinstance(ax, Axes3D):
        ax.set_zlabel("Add a label / unit" if zlabel is None else zlabel)


def _annotate_colormap(
    df: pd.DataFrame,
    ax: Axes,
    sci_annotation_bounds: tuple[float, float],
    annotation_style: str,
    hm: QuadMesh,
) -> None:
    hm.update_scalarmappable()  # So that get_facecolor is an array
    xpos, ypos = np.meshgrid(
        np.arange(len(df.columns)),
        np.arange(len(df.index)),
    )
    for x, y, val, color in zip(
        xpos.flat,
        ypos.flat,
        hm.get_array().flat,  # type: ignore
        hm.get_facecolor(),
        strict=True,
    ):
        if sci_annotation_bounds[0] < abs(val) <= sci_annotation_bounds[1]:
            val_text = f"{val:.{annotation_style}}"
        else:
            val_text = f"{val:.0e}"
        ax.text(
            x + 0.5,
            y + 0.5,
            val_text,
            ha="center",
            va="center",
            color="black" if _relative_luminance(color) > 0.45 else "white",  # type: ignore  # noqa: PLR2004
        )


def add_grid(ax: Axes) -> Axes:
    ax.grid(visible=True)
    ax.set_axisbelow(b=True)
    return ax


##########################################################################
# General plot layout
##########################################################################


def _default_fig_ax(
    *,
    ax: Axes | None,
    grid: bool,
    figsize: tuple[float, float] | None = None,
) -> FigAx:
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    if grid:
        add_grid(ax)
    return fig, ax


def _default_fig_axs(
    axs: list[Axes] | None,
    *,
    ncols: int,
    nrows: int,
    figsize: tuple[float, float] | None,
    sharex: bool,
    sharey: bool,
    grid: bool,
) -> FigAxs:
    if axs is None or len(axs) == 0:
        fig, axs_array = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            squeeze=False,
            layout="constrained",
        )
        axs = list(axs_array.flatten())
    else:
        fig = cast(Figure, axs[0].get_figure())

    if grid:
        for ax in axs:
            add_grid(ax)
    return fig, axs


def two_axes(
    *,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = False,
) -> FigAxs:
    return _default_fig_axs(
        None,
        ncols=2,
        nrows=1,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )


def grid_layout(
    n_groups: int,
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> tuple[Figure, list[Axes]]:
    n_cols = min(n_groups, n_cols)
    n_rows = math.ceil(n_groups / n_cols)
    figsize = (n_cols * col_width, n_rows * row_height)

    return _default_fig_axs(
        None,
        ncols=n_cols,
        nrows=n_rows,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )


##########################################################################
# Plots
##########################################################################


def lines(
    x: pd.DataFrame,
    *,
    ax: Axes | None = None,
    grid: bool = True,
) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    ax.plot(x)
    ax.legend(x.columns)
    return fig, ax


def lines_grouped(
    groups: list[pd.DataFrame] | list[pd.Series],
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    fig, axs = grid_layout(
        len(groups),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )

    for group, ax in zip(groups, axs, strict=False):
        group.plot(ax=ax)

    for i in range(len(groups), len(axs)):
        axs[i].set_visible(False)

    return fig, axs


def line_autogrouped(
    s: pd.Series | pd.DataFrame,
    *,
    n_cols: int = 2,
    col_width: float = 4,
    row_height: float = 3,
    max_group_size: int = 6,
    grid: bool = True,
) -> FigAxs:
    if isinstance(s, pd.Series):
        group_names = _partition_by_order_of_magnitude(s)
    else:
        group_names = _partition_by_order_of_magnitude(s.max())

    group_names = _split_large_groups(group_names, max_size=max_group_size)

    groups: list[pd.Series] | list[pd.DataFrame]

    if isinstance(s, pd.Series):
        groups = [s.loc[group] for group in group_names]
    else:
        groups = [s.loc[:, group] for group in group_names]

    return lines_grouped(
        groups,
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        grid=grid,
    )


def line_mean_std(
    df: pd.DataFrame,
    *,
    label: str | None = None,
    ax: Axes | None = None,
    color: str | None = None,
    alpha: float = 0.2,
    grid: bool = True,
) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    color = _default_color(ax=ax, color=color)

    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ax.plot(
        mean,
        color=color,
        label=label,
    )
    ax.fill_between(
        df.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )
    return fig, ax


def lines_mean_std_from_2d_idx(
    df: pd.DataFrame,
    *,
    names: list[str] | None = None,
    ax: Axes | None = None,
    alpha: float = 0.2,
    grid: bool = True,
) -> FigAx:
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, ax = _default_fig_ax(ax=ax, grid=grid)

    for name in df.columns if names is None else names:
        line_mean_std(
            df[name].unstack().T,
            label=name,
            alpha=alpha,
            ax=ax,
        )
    ax.legend()
    return fig, ax


def heatmap(
    df: pd.DataFrame,
    *,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize | None = None,
    ax: Axes | None = None,
    cax: Axes | None = None,
    sci_annotation_bounds: tuple[float, float] = (0.01, 100),
    annotation_style: str = "2g",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = _default_fig_ax(
        ax=ax,
        figsize=(
            1.5 * len(df.index),
            1.5 * len(df.columns),
        ),
        grid=False,
    )
    if norm is None:
        norm = _norm_with_zero_center(df)

    hm = ax.pcolormesh(df.T, norm=norm, cmap=cmap)
    ax.set_xticks(
        np.arange(0, len(df.index), 1) + 0.5,
        labels=df.index,
    )
    ax.set_yticks(
        np.arange(0, len(df.columns), 1) + 0.5,
        labels=df.columns,
    )

    if annotate:
        _annotate_colormap(df, ax, sci_annotation_bounds, annotation_style, hm)

    if colorbar:
        # Add a colorbar
        cb = fig.colorbar(hm, cax, ax)
        cb.outline.set_linewidth(0)  # type: ignore

    if invert_yaxis:
        ax.invert_yaxis()
    return fig, ax, hm


def heatmap_from_2d_idx(
    df: pd.DataFrame,
    variable: str,
    ax: Axes | None = None,
) -> FigAx:
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, ax = _default_fig_ax(ax=ax, grid=False)
    df2d = df[variable].unstack()

    ax.set_title(variable)
    # Note: pcolormesh swaps index/columns
    hm = ax.pcolormesh(df2d.T)
    ax.set_xlabel(df2d.index.name)
    ax.set_ylabel(df2d.columns.name)
    ax.set_xticks(
        np.arange(0, len(df2d.index), 1) + 0.5,
        labels=[f"{i:.2f}" for i in df2d.index],
    )
    ax.set_yticks(
        np.arange(0, len(df2d.columns), 1) + 0.5,
        labels=[f"{i:.2f}" for i in df2d.columns],
    )

    # Add colorbar
    fig.colorbar(hm, ax=ax)
    return fig, ax


def heatmaps_from_2d_idx(
    df: pd.DataFrame,
    *,
    n_cols: int = 3,
    col_width_factor: float = 1,
    row_height_factor: float = 0.6,
    sharex: bool = True,
    sharey: bool = False,
) -> FigAxs:
    idx = cast(pd.MultiIndex, df.index)

    fig, axs = grid_layout(
        n_groups=len(df.columns),
        n_cols=min(n_cols, len(df)),
        col_width=len(idx.levels[0]) * col_width_factor,
        row_height=len(idx.levels[1]) * row_height_factor,
        sharex=sharex,
        sharey=sharey,
        grid=False,
    )
    for ax, var in zip(axs, df.columns, strict=False):
        heatmap_from_2d_idx(df, var, ax=ax)
    return fig, axs


def violins(
    df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    grid: bool = True,
) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    sns.violinplot(df, ax=ax)
    return fig, ax


def violins_from_2d_idx(
    df: pd.DataFrame,
    *,
    grid: bool = True,
) -> FigAxs:
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, axs = grid_layout(
        len(df.columns),
        n_cols=4,
        row_height=2,
        sharex=True,
        grid=grid,
    )

    for ax, col in zip(axs[: len(df.columns)], df.columns, strict=True):
        ax.set_ylabel(col)
        sns.violinplot(df[col].unstack(), ax=ax)

    for ax in axs[len(df.columns) :]:
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0)
        ax.yaxis.set_ticks([])

    for ax in axs:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
    return fig, axs


def shade_protocol(
    protocol: pd.Series,
    *,
    ax: Axes,
    cmap_name: str = "Greys_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.5,
    add_legend: bool = True,
) -> None:
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.legend import Legend
    from matplotlib.patches import Patch

    cmap = colormaps[cmap_name]
    norm = Normalize(
        vmin=default_if_none(vmin, protocol.min()),
        vmax=default_if_none(vmax, protocol.max()),
    )

    t0 = pd.Timedelta(seconds=0)
    for t_end, val in protocol.items():
        t_end = cast(pd.Timedelta, t_end)
        ax.axvspan(
            t0.total_seconds(),
            t_end.total_seconds(),
            facecolor=cmap(norm(val)),
            edgecolor=None,
            alpha=alpha,
        )
        t0 = t_end  # type: ignore

    if add_legend:
        ax.add_artist(
            Legend(
                ax,
                handles=[
                    Patch(
                        facecolor=cmap(norm(val)),
                        alpha=alpha,
                        label=val,
                    )  # type: ignore
                    for val in protocol
                ],
                labels=protocol,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.0),
                title=default_if_none(cast(str, protocol.name), "protocol"),
            )
        )


##########################################################################
# Label Plots
##########################################################################


def relative_label_distribution(
    mapper: LabelMapper | LinearLabelMapper,
    concs: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 3,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    variables = list(mapper.label_variables) if subset is None else subset
    fig, axs = grid_layout(
        n_groups=len(variables),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharey=sharey,
        grid=grid,
    )
    if isinstance(mapper, LabelMapper):
        for ax, name in zip(axs, variables, strict=False):
            for i in range(mapper.label_variables[name]):
                isos = mapper.get_isotopomers_of_at_position(name, i)
                labels = cast(pd.DataFrame, concs.loc[:, isos])
                total = concs.loc[:, f"{name}__total"]
                (labels.sum(axis=1) / total).plot(ax=ax, label=f"C{i+1}")
            ax.set_title(name)
            ax.legend()
    else:
        for ax, (name, isos) in zip(
            axs, mapper.get_isotopomers(variables).items(), strict=False
        ):
            concs.loc[:, isos].plot(ax=ax)
            ax.set_title(name)
            ax.legend([f"C{i+1}" for i in range(len(isos))])

    return fig, axs
