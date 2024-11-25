import itertools as it
import math
from typing import cast

import numpy as np
import pandas as pd
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


def relative_luminance(color: Array) -> float:
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


##########################################################################
# General plot layout
##########################################################################


def _default_fig_ax(
    ax: Axes | None,
    figsize: tuple[float, float] | None = None,
) -> FigAx:
    if ax is None:
        return plt.subplots(nrows=1, ncols=1, figsize=figsize)
    return cast(Figure, ax.get_figure()), ax


def _default_fig_axs(
    axs: list[Axes] | None,
    *,
    ncols: int,
    nrows: int,
    sharex: bool,
    sharey: bool,
) -> FigAxs:
    if axs is None or len(axs) == 0:
        return plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
        )
    return cast(Figure, axs[0].get_figure()), axs


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


def grid_layout(
    n_groups: int,
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharex: bool = True,
    sharey: bool = False,
) -> tuple[Figure, list[Axes]]:
    n_cols = min(n_groups, n_cols)
    n_rows = math.ceil(n_groups / n_cols)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * col_width, n_rows * row_height),
        sharex=sharex,
        sharey=sharey,
        layout="constrained",
        squeeze=False,
    )
    return fig, list(axs.flatten())


##########################################################################
# Small customisation
##########################################################################


def add_grid(ax: Axes) -> Axes:
    ax.grid(visible=True)
    ax.set_axisbelow(b=True)
    return ax


##########################################################################
# Plots
##########################################################################


def line(x: pd.DataFrame, *, ax: Axes | None = None) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax)
    x.plot(ax=ax)
    return fig, ax


def line_grouped(
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
    )

    for group, ax in zip(groups, axs, strict=False):
        group.plot(ax=ax, grid=grid)

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

    return line_grouped(
        groups,
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        grid=grid,
    )


def line_mean_std(
    df: pd.DataFrame,
    *,
    label: str | None,
    ax: Axes | None = None,
    color: str | None = None,
    alpha: float = 0.2,
) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax)
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


def mc_line_mean_std(
    df: pd.DataFrame,
    var: str,
    label: str | None = None,
    color: str | None = None,
    ax: Axes | None = None,
) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax)
    line_mean_std(
        ax=ax,
        df=df[var].unstack().T,
        color=color,
        label=var if label is None else label,
    )
    return fig, ax


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
            color="black" if relative_luminance(color) > 0.45 else "white",  # type: ignore  # noqa: PLR2004
        )


def heatmap(
    df: pd.DataFrame,
    *,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = True,
    cmap: str = "viridis",
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
    )
    if norm is None:
        norm = _norm_with_zero_center(df)

    hm = ax.pcolormesh(df.T, cmap=cmap)
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


def heatmap_from_2d_idx(df: pd.DataFrame, variable: str) -> None:
    df2d = df[variable].unstack()

    fig, ax = plt.subplots()
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
) -> FigAxs:
    variables = list(mapper.label_variables) if subset is None else subset
    fig, axs = grid_layout(
        n_groups=len(variables),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharey=sharey,
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


# def plot_derived(
#     self,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     normalise: float | ArrayLike | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     compounds = self.model.get_derived_compounds()
#     y = cast(
#         pd.DataFrame,
#         self.get_full_results(normalise=normalise, concatenated=True),
#     )
#     if y is None:
#         return None, None
#     return plot(
#         plot_args=(y.loc[:, compounds],),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_all(
#     self,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     normalise: float | ArrayLike | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     compounds = self.model.get_all_compounds()
#     y = cast(
#         pd.DataFrame,
#         self.get_full_results(normalise=normalise, concatenated=True),
#     )
#     if y is None:
#         return None, None
#     return plot(
#         plot_args=(y.loc[:, compounds],),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_selection(
#     self,
#     *,
#     compounds: list[str],
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     normalise: float | ArrayLike | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     y = cast(
#         pd.DataFrame,
#         self.get_full_results(normalise=normalise, concatenated=True),
#     )
#     if y is None:
#         return None, None
#     return plot(
#         plot_args=(y.loc[:, compounds],),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_grid(
#     self,
#     compound_groups: list[list[str]],
#     *,
#     ncols: int | None = None,
#     sharex: bool = True,
#     sharey: bool = True,
#     xlabels: str | Iterable[str] | None = None,
#     ylabels: str | Iterable[str] | None = None,
#     normalise: float | ArrayLike | None = None,
#     plot_titles: Iterable[str] | None = None,
#     figure_title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axes | None]:
#     """Plot simulation results of the compound groups as a grid.

#     Examples
#     --------
#     >>> plot_grid([["x1", "x2"], ["x3", "x4]])

#     """
#     y = cast(
#         pd.DataFrame,
#         self.get_full_results(normalise=normalise, concatenated=True),
#     )
#     if y is None:
#         return None, None
#     plot_groups = [(y.loc[:, compounds],) for compounds in compound_groups]
#     return plot_grid(
#         plot_groups=plot_groups,  # type: ignore
#         legend_groups=compound_groups,
#         ncols=ncols,
#         sharex=sharex,
#         sharey=sharey,
#         xlabels=xlabels,
#         ylabels=ylabels,
#         figure_title=figure_title,
#         plot_titles=plot_titles,
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_derivatives(
#     self,
#     compounds: list[str],
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     rhs = self.get_right_hand_side(annotate_names=False)
#     if len(rhs) == 0:
#         return None, None

#     return plot(
#         plot_args=(rhs.loc[:, compounds],),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_against_variable(
#     self,
#     variable: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if xlabel is None:
#         xlabel = variable
#     results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
#     if results is None:
#         return None, None
#     compounds = cast(list[str], self.model.get_compounds())
#     x = results.loc[:, variable].values  # type: ignore
#     y = results.loc[:, compounds].values
#     return plot(
#         plot_args=(x, y),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_derived_against_variable(
#     self,
#     variable: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if xlabel is None:
#         xlabel = variable
#     results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
#     if results is None:
#         return None, None
#     compounds = cast(list[str], self.model.get_derived_compounds())
#     x = results.loc[:, variable].values  # type: ignore
#     y = results.loc[:, compounds].values
#     return plot(
#         plot_args=(x, y),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_all_against_variable(
#     self,
#     variable: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if xlabel is None:
#         xlabel = variable
#     results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
#     if results is None:
#         return None, None
#     compounds = cast(list[str], self.model.get_all_compounds())
#     x = results.loc[:, variable].values  # type: ignore
#     y = results.loc[:, compounds].values
#     return plot(
#         plot_args=(x, y),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_selection_against_variable(
#     self,
#     compounds: Iterable[str],
#     variable: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if xlabel is None:
#         xlabel = variable
#     results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
#     if results is None:
#         return None, None
#     x = results.loc[:, variable].values  # type: ignore
#     y = results.loc[:, compounds].values  # type: ignore
#     return plot(
#         plot_args=(x, y),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_fluxes(
#     self,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     normalise: float | ArrayLike | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     y = self.get_fluxes(normalise=normalise, concatenated=True)
#     if y is None:
#         return None, None
#     return plot(
#         plot_args=(y,),
#         legend=y.columns,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_flux_selection(
#     self,
#     rate_names: list[str],
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     normalise: float | ArrayLike | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     y = self.get_fluxes(normalise=normalise, concatenated=True)
#     if y is None:
#         return None, None
#     y = cast(pd.DataFrame, y)
#     return plot(
#         plot_args=(y.loc[:, rate_names],),
#         legend=rate_names,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_fluxes_grid(
#     self,
#     rate_groups: list[list[str]],
#     *,
#     ncols: int | None = None,
#     sharex: bool = True,
#     sharey: bool = True,
#     xlabels: list[str] | None = None,
#     ylabels: list[str] | None = None,
#     normalise: float | ArrayLike | None = None,
#     plot_titles: Iterable[str] | None = None,
#     figure_title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Array | None]:
#     """Plot simulation results of the compound groups as a grid.

#     Examples
#     --------
#     >>> plot_fluxes_grid([["v1", "v2"], ["v3", "v4]])

#     """
#     fluxes = self.get_fluxes(normalise=normalise, concatenated=True)
#     if fluxes is None:
#         return None, None
#     fluxes = cast(pd.DataFrame, fluxes)
#     plot_groups = [(cast(Array, fluxes.loc[:, group]),) for group in rate_groups]
#     return plot_grid(
#         plot_groups=plot_groups,  # type: ignore
#         legend_groups=rate_groups,
#         ncols=ncols,
#         sharex=sharex,
#         sharey=sharey,
#         xlabels=xlabels,
#         ylabels=ylabels,
#         figure_title=figure_title,
#         plot_titles=plot_titles,
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_fluxes_against_variable(
#     self,
#     variable: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if (c := self.get_results()) is None:
#         return None, None
#     if (v := self.get_fluxes(concatenated=True)) is None:
#         return None, None
#     rate_names = self.model.get_rate_names()
#     x = c.loc[:, variable]
#     y = v.loc[:, rate_names].to_numpy()
#     return plot(
#         plot_args=(x, y),
#         legend=rate_names,
#         xlabel=variable if xlabel is None else xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_flux_selection_against_variable(
#     self,
#     rate_names: list[str],
#     variable: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if (c := self.get_results()) is None:
#         return None, None
#     if (v := self.get_fluxes()) is None:
#         return None, None
#     x = c.loc[:, variable]
#     y = v.loc[:, rate_names].to_numpy()
#     return plot(
#         plot_args=(x, y),
#         legend=rate_names,
#         xlabel=variable if xlabel is None else xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_phase_plane(
#     self,
#     cpd1: str,
#     cpd2: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     if (c := self.get_results()) is None:
#         return None, None

#     return plot(
#         plot_args=(c.loc[:, cpd1], c.loc[:, cpd2]),
#         legend=None,
#         xlabel=cpd1 if xlabel is None else xlabel,
#         ylabel=cpd2 if ylabel is None else ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_phase_space(
#     self,
#     cpd1: str,
#     cpd2: str,
#     cpd3: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     zlabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     kwargs = _get_plot_kwargs(
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#         legend_kwargs=legend_kwargs,
#     )
#     kwargs["subplot"].update({"projection": "3d"})

#     if (c := self.get_results()) is None:
#         return None, None

#     x = c.loc[:, cpd1]
#     y = c.loc[:, cpd2]
#     z = c.loc[:, cpd3]

#     if x is None or y is None or z is None:
#         return None, None

#     xlabel = cpd1 if xlabel is None else xlabel
#     ylabel = cpd2 if ylabel is None else ylabel
#     zlabel = cpd3 if zlabel is None else zlabel

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"])
#     else:
#         fig = ax.get_figure()
#     fig = cast(Figure, fig)
#     ax = cast(Axis, ax)

#     ax.plot(
#         x,
#         y,
#         z,
#         **kwargs["plot"],
#     )
#     _style_subplot(
#         ax=ax,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         zlabel=zlabel,
#         title=title,
#         grid=grid,
#         kwargs=kwargs,
#     )
#     if tight_layout:
#         fig.tight_layout()
#     return fig, ax


# def plot_trajectories(
#     self,
#     cpd1: str,
#     cpd2: str,
#     *,
#     cpd1_bounds: tuple[float, float],
#     cpd2_bounds: tuple[float, float],
#     n: int,
#     y0: dict[str, float],
#     t0: float = 0,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     xlabel = cpd1 if xlabel is None else xlabel
#     ylabel = cpd2 if ylabel is None else ylabel

#     kwargs = _get_plot_kwargs(
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#         legend_kwargs=legend_kwargs,
#     )

#     x = np.linspace(*cpd1_bounds, n)
#     y = np.linspace(*cpd2_bounds, n)
#     u = np.zeros((n, n))
#     v = np.zeros((n, n))

#     fcd = self.model.get_full_concentration_dict(y=y0, t=t0)
#     for i, s1 in enumerate(x):
#         for j, s2 in enumerate(y):
#             # Update y0 to new values
#             fcd.update({cpd1: s1, cpd2: s2})
#             rhs = self.model.get_right_hand_side(y=fcd, t=t0)
#             u[i, j] = rhs[f"d{cpd1}dt"]
#             v[i, j] = rhs[f"d{cpd2}dt"]

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"])
#     else:
#         fig = ax.get_figure()
#     fig = cast(Figure, fig)
#     ax = cast(Axis, ax)

#     ax.quiver(x, y, u.T, v.T)
#     _style_subplot(
#         ax=ax,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         kwargs=kwargs,
#     )
#     if tight_layout:
#         fig.tight_layout()
#     return fig, ax


# def plot_3d_trajectories(
#     self,
#     cpd1: str,
#     cpd2: str,
#     cpd3: str,
#     *,
#     cpd1_bounds: tuple[float, float],
#     cpd2_bounds: tuple[float, float],
#     cpd3_bounds: tuple[float, float],
#     n: int,
#     y0: dict[str, float],
#     t0: float = 0,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     zlabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     kwargs = _get_plot_kwargs(
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#         legend_kwargs=legend_kwargs,
#     )
#     kwargs["subplot"].update({"projection": "3d"})

#     x = np.linspace(*cpd1_bounds, n)
#     y = np.linspace(*cpd2_bounds, n)
#     z = np.linspace(*cpd3_bounds, n)
#     u = np.zeros((n, n, n))
#     v = np.zeros((n, n, n))
#     w = np.zeros((n, n, n))

#     fcd = self.model.get_full_concentration_dict(y=y0, t=t0)
#     for i, s1 in enumerate(x):
#         for j, s2 in enumerate(y):
#             for k, s3 in enumerate(y):
#                 fcd.update({cpd1: s1, cpd2: s2, cpd3: s3})
#                 rhs = self.model.get_right_hand_side(y=fcd, t=t0)
#                 u[i, j, k] = rhs[f"d{cpd1}dt"]
#                 v[i, j, k] = rhs[f"d{cpd2}dt"]
#                 w[i, j, k] = rhs[f"d{cpd3}dt"]

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"])
#     else:
#         fig = ax.get_figure()
#     fig = cast(Figure, fig)
#     ax = cast(Axis, ax)

#     X, Y, Z = np.meshgrid(x, y, z)
#     ax.quiver(
#         X,
#         Y,
#         Z,
#         np.transpose(u, [1, 0, 2]),
#         np.transpose(v, [1, 0, 2]),
#         np.transpose(w, [1, 0, 2]),
#         length=0.05,
#         normalize=True,
#         alpha=0.5,
#     )
#     xlabel = cpd1 if xlabel is None else xlabel
#     ylabel = cpd2 if ylabel is None else ylabel
#     zlabel = cpd3 if zlabel is None else zlabel
#     _style_subplot(
#         ax=ax,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         zlabel=zlabel,
#         title=title,
#         grid=grid,
#         kwargs=kwargs,
#     )
#     if tight_layout:
#         fig.tight_layout()
#     return fig, ax


# def plot(
#     self,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     """Plot all total concentrations."""
#     compounds = sorted(
#         [f"{i}__total" for i in self.model.label_compounds]
#         + self.model.nonlabel_compounds
#     )
#     y = self.get_full_results(concatenated=True, include_readouts=False)
#     if y is None:
#         return None, None
#     y = y.loc[:, compounds]
#     return plot(
#         plot_args=(y,),
#         legend=None,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def _calculate_label_distribution(
#     self, *, compound: str, relative: bool
# ) -> Array | None:
#     """Calculate the label distribution of a compound."""
#     total_concentration = self.get_total_concentration(compound=compound)
#     if total_concentration is None:
#         return None
#     concentrations = []
#     for position in range(
#         self.model.get_compound_number_of_label_positions(compound=compound)
#     ):
#         concentration = self.get_concentration_at_positions(
#             compound=compound, positions=position
#         )
#         if concentration is None:
#             return None
#         if relative:
#             concentration = concentration / total_concentration
#         concentrations.append(concentration)
#     return np.array(concentrations).T


# def plot_label_distribution(
#     self,
#     compound: str,
#     relative: bool = True,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     prefix: str = "Pos ",
#     initial_index: int = 0,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     """Plot label distribution of a compound."""
#     if ylabel is None and relative:
#         ylabel = "Relative concentration"
#     x = self.get_time()
#     y = self._calculate_label_distribution(compound=compound, relative=relative)
#     if x is None or y is None:
#         return None, None

#     return plot(
#         plot_args=(x, y),
#         legend=self._make_legend_labels(
#             prefix=prefix, compound=compound, initial_index=initial_index
#         ),
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_label_distribution_grid(
#     self,
#     compounds: list[str],
#     relative: bool = True,
#     ncols: int | None = None,
#     sharex: bool = True,
#     sharey: bool = True,
#     xlabels: str | list[str] | None = None,
#     ylabels: str | list[str] | None = None,
#     plot_titles: Iterable[str] | None = None,
#     figure_title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     prefix: str = "Pos ",
#     initial_index: int = 0,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Array | None]:
#     """Plot label distributions of multiple compounds on a grid."""
#     time = self.get_time()
#     if time is None:
#         return None, None
#     plot_groups = [
#         (
#             time,
#             self._calculate_label_distribution(compound=compound, relative=relative),
#         )
#         for compound in compounds
#     ]
#     legend_groups = [
#         self._make_legend_labels(
#             prefix=prefix, compound=compound, initial_index=initial_index
#         )
#         for compound in compounds
#     ]
#     if ylabels is None and relative:
#         ylabels = "Relative concentration"
#     if plot_titles is None:
#         plot_titles = compounds
#     return plot_grid(
#         plot_groups=plot_groups,  # type: ignore
#         legend_groups=legend_groups,
#         ncols=ncols,
#         sharex=sharex,
#         sharey=sharey,
#         xlabels=xlabels,
#         ylabels=ylabels,
#         figure_title=figure_title,
#         plot_titles=plot_titles,
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_all_label_distributions(
#     self,
#     relative: bool = True,
#     ncols: int | None = None,
#     sharex: bool = True,
#     sharey: bool = True,
#     xlabels: str | list[str] | None = None,
#     ylabels: str | list[str] | None = None,
#     plot_titles: Iterable[str] | None = None,
#     figure_title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     prefix: str = "Pos ",
#     initial_index: int = 0,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Array | None]:
#     """Plot label distributions of all compounds on a grid."""
#     time = self.get_time()
#     if time is None:
#         return None, None
#     compounds = self.model.label_compounds
#     plot_groups = [
#         (
#             time,
#             self._calculate_label_distribution(compound=compound, relative=relative),
#         )
#         for compound in compounds
#     ]
#     legend_groups = [
#         self._make_legend_labels(
#             prefix=prefix, compound=compound, initial_index=initial_index
#         )
#         for compound in compounds
#     ]
#     if ylabels is None and relative:
#         ylabels = "Relative concentration"

#     return plot_grid(
#         plot_groups=plot_groups,  # type: ignore
#         legend_groups=legend_groups,
#         ncols=ncols,
#         sharex=sharex,
#         sharey=sharey,
#         xlabels=xlabels,
#         ylabels=ylabels,
#         figure_title=figure_title,
#         plot_titles=plot_titles,
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# # Linear label models


# def plot_label_distribution(
#     self,
#     compound: str,
#     relative: bool = True,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
#     legend_prefix: str = "Pos ",
#     initial_index: int = 0,
# ) -> tuple[Figure, Axis]:
#     """Plot label distribution of a compound."""
#     if ylabel is None and relative:
#         ylabel = "Relative concentration"
#     x = self.get_time()
#     y = self.get_label_distribution(compound=compound)
#     legend = self._make_legend_labels(legend_prefix, compound, initial_index)
#     if title is None:
#         title = compound
#     return plot(
#         plot_args=(x, y),
#         legend=legend,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_label_distribution_grid(
#     self,
#     compounds: list[str],
#     relative: bool = True,
#     ncols: int | None = None,
#     sharex: bool = True,
#     sharey: bool = True,
#     xlabels: str | list[str] | None = None,
#     ylabels: str | list[str] | None = None,
#     plot_titles: list[str] | None = None,
#     figure_title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
#     legend_prefix: str = "Pos ",
#     initial_index: int = 0,
# ) -> tuple[Figure | None, Array | None]:
#     """Plot label distributions of multiple compounds on a grid."""
#     time = self.get_time()
#     plot_groups = [
#         (time, self.get_label_distribution(compound=compound)) for compound in compounds
#     ]
#     legend_groups = [
#         self._make_legend_labels(legend_prefix, compound, initial_index)
#         for compound in compounds
#     ]
#     if ylabels is None and relative:
#         ylabels = "Relative concentration"
#     if plot_titles is None:
#         plot_titles = compounds
#     return plot_grid(
#         plot_groups=plot_groups,  # type: ignore
#         legend_groups=legend_groups,
#         ncols=ncols,
#         sharex=sharex,
#         sharey=sharey,
#         xlabels=xlabels,
#         ylabels=ylabels,
#         figure_title=figure_title,
#         plot_titles=plot_titles,
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_all_label_distributions(
#     self,
#     relative: bool = True,
#     ncols: int | None = None,
#     sharex: bool = True,
#     sharey: bool = True,
#     xlabels: str | list[str] | None = None,
#     ylabels: str | list[str] | None = None,
#     plot_titles: Iterable[str] | None = None,
#     figure_title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
#     legend_prefix: str = "Pos ",
#     initial_index: int = 0,
# ) -> tuple[Figure | None, Array | None]:
#     """Plot label distributions of all compounds on a grid."""
#     time = self.get_time()
#     compounds = self.model.isotopomers
#     plot_groups = [
#         (time, self.get_label_distribution(compound=compound)) for compound in compounds
#     ]
#     legend_groups = [
#         self._make_legend_labels(legend_prefix, compound, initial_index)
#         for compound in compounds
#     ]
#     if ylabels is None and relative:
#         ylabels = "Relative concentration"
#     if plot_titles is None:
#         plot_titles = compounds

#     return plot_grid(
#         plot_groups=plot_groups,  # type: ignore
#         legend_groups=legend_groups,
#         ncols=ncols,
#         sharex=sharex,
#         sharey=sharey,
#         xlabels=xlabels,
#         ylabels=ylabels,
#         figure_title=figure_title,
#         plot_titles=plot_titles,
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# # Simulator


# def plot(
#     self,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     normalise: float | ArrayLike | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     ax: Axis | None = None,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any | None] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[Figure | None, Axis | None]:
#     """Plot simulation results for a selection of compounds."""
#     compounds = self.model.get_compounds()
#     y = self.get_full_results(normalise=normalise, concatenated=True)
#     if y is None:
#         return None, None
#     return plot(
#         plot_args=(y.loc[:, compounds],),
#         legend=compounds,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         title=title,
#         grid=grid,
#         tight_layout=tight_layout,
#         ax=ax,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# def plot_producing_and_consuming(
#     self,
#     compound: str,
#     *,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     title: str | None = None,
#     grid: bool = True,
#     tight_layout: bool = True,
#     figure_kwargs: dict[str, Any] | None = None,
#     subplot_kwargs: dict[str, Any] | None = None,
#     plot_kwargs: dict[str, Any] | None = None,
#     grid_kwargs: dict[str, Any] | None = None,
#     legend_kwargs: dict[str, Any] | None = None,
#     tick_kwargs: dict[str, Any | None] | None = None,
#     label_kwargs: dict[str, Any] | None = None,
#     title_kwargs: dict[str, Any] | None = None,
# ) -> tuple[None, None] | tuple[Figure, Axes]:
#     producing: list[Array] = []
#     consuming: list[Array] = []
#     producing_names: list[str] = []
#     consuming_names: list[str] = []

#     if (v := self.get_fluxes()) is None:
#         return None, None

#     title = compound if title is None else title

#     for rate_name, factor in self.model.stoichiometries_by_compounds[compound].items():
#         if factor > 0:
#             producing.append(v[rate_name].to_numpy() * factor)  # type: ignore
#             producing_names.append(rate_name)
#         else:
#             consuming.append(v[rate_name].to_numpy() * -factor)  # type: ignore
#             consuming_names.append(rate_name)

#     return plot_grid(
#         plot_groups=[
#             (v.index.to_numpy(), np.array(producing).T),
#             (v.index.to_numpy(), np.array(consuming).T),
#         ],
#         legend_groups=[producing_names, consuming_names],
#         xlabels=xlabel,
#         ylabels=ylabel,
#         figure_title=title,
#         plot_titles=("Producing", "Consuming"),
#         grid=grid,
#         tight_layout=tight_layout,
#         figure_kwargs=figure_kwargs,
#         subplot_kwargs=subplot_kwargs,
#         plot_kwargs=plot_kwargs,
#         grid_kwargs=grid_kwargs,
#         legend_kwargs=legend_kwargs,
#         tick_kwargs=tick_kwargs,
#         label_kwargs=label_kwargs,
#         title_kwargs=title_kwargs,
#     )


# from modelbase2.types import Array, ArrayLike, Axes, Axis

# if TYPE_CHECKING:
#     from collections.abc import Iterable

#     import pandas as pd
#     from matplotlib.collections import QuadMesh
#     from matplotlib.figure import Figure
