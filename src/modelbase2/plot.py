from typing import cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from modelbase2.types import default_if_none

type FigAx = tuple[Figure, Axes]
type FigAxs = tuple[Figure, list[Axes]]


def _default_fig_ax(ax: Axes | None) -> FigAx:
    if ax is None:
        return plt.subplots(nrows=1, ncols=1)
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


def add_grid(ax: Axes) -> Axes:
    ax.grid(visible=True)
    ax.set_axisbelow(b=True)
    return ax


def line(x: pd.DataFrame, *, ax: Axes | None = None) -> FigAx:
    fig, ax = _default_fig_ax(ax=ax)
    x.plot(ax=ax)
    return fig, ax


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


def _plot_line_median_std(
    ax: Axes,
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

# from matplotlib.colors import (
#     LogNorm,
#     Normalize,
#     SymLogNorm,
#     colorConverter,  # type: ignore
# )

# from modelbase2.types import Array, ArrayLike, Axes, Axis

# if TYPE_CHECKING:
#     from collections.abc import Iterable

#     import pandas as pd
#     from matplotlib.collections import QuadMesh
#     from matplotlib.figure import Figure

# def relative_luminance(color: ArrayLike) -> float:
#     """Calculate the relative luminance of a color."""
#     rgb = colorConverter.to_rgba_array(color)[:, :3]

#     # If RsRGB <= 0.03928 then R = RsRGB/12.92 else R = ((RsRGB+0.055)/1.055) ^ 2.4
#     rsrgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

#     # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
#     rel_luminance: ArrayLike = np.matmul(rsrgb, [0.2126, 0.7152, 0.0722])
#     return rel_luminance[0]


# def get_norm(vmin: float, vmax: float) -> plt.Normalize:
#     if vmax < 1000 and vmin > -1000:
#         norm = Normalize(vmin=vmin, vmax=vmax)
#     elif vmin <= 0:
#         norm = SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, base=10)
#     else:
#         norm = LogNorm(vmin=vmin, vmax=vmax)
#     return norm


# def heatmap_from_dataframe(
#     df: pd.DataFrame,
#     title: str | None = None,
#     xlabel: str | None = None,
#     ylabel: str | None = None,
#     annotate: bool = True,
#     colorbar: bool = True,
#     cmap: str = "viridis",
#     vmax: float | None = None,
#     vmin: float | None = None,
#     norm: plt.Normalize | None = None,
#     ax: Axis | None = None,
#     cax: Axis | None = None,
#     sci_annotation_bounds: tuple[float, float] = (0.01, 100),
#     annotation_style: str = "2g",
# ) -> tuple[Figure, Axis, QuadMesh]:
#     data = df.values
#     rows = df.index
#     columns = df.columns

#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure()

#     # Create norm
#     if norm is None:
#         if vmax is None:
#             vmax = np.nanmax(data)
#         if vmin is None:
#             vmin = np.nanmin(data)
#         vmax = cast(float, vmax)
#         vmin = cast(float, vmin)
#         norm = get_norm(vmin=vmin, vmax=vmax)

#     # Create heatmap
#     hm = ax.pcolormesh(data, norm=norm, cmap=cmap)

#     # Despine axis
#     for side in ["top", "right", "left", "bottom"]:
#         ax.spines[side].set_visible(False)

#     # Set the axis limits
#     ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))

#     # Set ticks and ticklabels
#     ax.set_xticks(np.arange(len(columns)) + 0.5)
#     ax.set_xticklabels(columns)

#     ax.set_yticks(np.arange(len(rows)) + 0.5)
#     ax.set_yticklabels(rows)

#     # Set title and axis labels
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     if annotate:
#         text_kwargs = {"ha": "center", "va": "center"}
#         hm.update_scalarmappable()  # So that get_facecolor is an array
#         xpos, ypos = np.meshgrid(np.arange(len(columns)), np.arange(len(rows)))
#         for x, y, val, color in zip(
#             xpos.flat, ypos.flat, hm.get_array().flat, hm.get_facecolor(), strict=True
#         ):
#             text_kwargs["color"] = (
#                 "black" if relative_luminance(color) > 0.45 else "white"
#             )
#             if sci_annotation_bounds[0] < abs(val) <= sci_annotation_bounds[1]:
#                 val_text = f"{val:.{annotation_style}}"
#             else:
#                 val_text = f"{val:.0e}"
#             ax.text(x + 0.5, y + 0.5, val_text, **text_kwargs)

#     if colorbar:
#         # Add a colorbar
#         cb = ax.figure.colorbar(hm, cax, ax)
#         cb.outline.set_linewidth(0)

#     # Invert the y axis to show the plot in matrix form
#     ax.invert_yaxis()
#     return fig, ax, hm
