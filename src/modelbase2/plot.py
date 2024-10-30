def plot_log(
    self,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    compounds = self.model.get_compounds()
    y = cast(
        pd.DataFrame,
        self.get_full_results(normalise=normalise, concatenated=True),
    )
    if y is None:
        return None, None
    fig, ax = plot(
        plot_args=(y.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig, ax


def plot_semilog(
    self,
    *,
    log_axis: str = "y",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    compounds = self.model.get_compounds()
    y = cast(
        pd.DataFrame,
        self.get_full_results(normalise=normalise, concatenated=True),
    )
    if y is None:
        return None, None
    fig, ax = plot(
        plot_args=(y.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )
    if log_axis == "y":
        ax.set_yscale("log")
    elif log_axis == "x":
        ax.set_xscale("log")
    else:
        msg = "log_axis must be either x or y"
        raise ValueError(msg)
    return fig, ax


def plot_derived(
    self,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    compounds = self.model.get_derived_compounds()
    y = cast(
        pd.DataFrame,
        self.get_full_results(normalise=normalise, concatenated=True),
    )
    if y is None:
        return None, None
    return plot(
        plot_args=(y.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_all(
    self,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    compounds = self.model.get_all_compounds()
    y = cast(
        pd.DataFrame,
        self.get_full_results(normalise=normalise, concatenated=True),
    )
    if y is None:
        return None, None
    return plot(
        plot_args=(y.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_selection(
    self,
    *,
    compounds: list[str],
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    y = cast(
        pd.DataFrame,
        self.get_full_results(normalise=normalise, concatenated=True),
    )
    if y is None:
        return None, None
    return plot(
        plot_args=(y.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_grid(
    self,
    compound_groups: list[list[str]],
    *,
    ncols: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    xlabels: str | Iterable[str] | None = None,
    ylabels: str | Iterable[str] | None = None,
    normalise: float | ArrayLike | None = None,
    plot_titles: Iterable[str] | None = None,
    figure_title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axes | None]:
    """Plot simulation results of the compound groups as a grid.

    Examples
    --------
    >>> plot_grid([["x1", "x2"], ["x3", "x4]])

    """
    y = cast(
        pd.DataFrame,
        self.get_full_results(normalise=normalise, concatenated=True),
    )
    if y is None:
        return None, None
    plot_groups = [(y.loc[:, compounds],) for compounds in compound_groups]
    return plot_grid(
        plot_groups=plot_groups,  # type: ignore
        legend_groups=compound_groups,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        xlabels=xlabels,
        ylabels=ylabels,
        figure_title=figure_title,
        plot_titles=plot_titles,
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_derivatives(
    self,
    compounds: list[str],
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    rhs = self.get_right_hand_side(annotate_names=False)
    if len(rhs) == 0:
        return None, None

    return plot(
        plot_args=(rhs.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_against_variable(
    self,
    variable: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if xlabel is None:
        xlabel = variable
    results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
    if results is None:
        return None, None
    compounds = cast(list[str], self.model.get_compounds())
    x = results.loc[:, variable].values  # type: ignore
    y = results.loc[:, compounds].values
    return plot(
        plot_args=(x, y),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_derived_against_variable(
    self,
    variable: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if xlabel is None:
        xlabel = variable
    results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
    if results is None:
        return None, None
    compounds = cast(list[str], self.model.get_derived_compounds())
    x = results.loc[:, variable].values  # type: ignore
    y = results.loc[:, compounds].values
    return plot(
        plot_args=(x, y),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_all_against_variable(
    self,
    variable: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if xlabel is None:
        xlabel = variable
    results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
    if results is None:
        return None, None
    compounds = cast(list[str], self.model.get_all_compounds())
    x = results.loc[:, variable].values  # type: ignore
    y = results.loc[:, compounds].values
    return plot(
        plot_args=(x, y),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_selection_against_variable(
    self,
    compounds: Iterable[str],
    variable: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if xlabel is None:
        xlabel = variable
    results = cast(pd.DataFrame, self.get_full_results(concatenated=True))
    if results is None:
        return None, None
    x = results.loc[:, variable].values  # type: ignore
    y = results.loc[:, compounds].values  # type: ignore
    return plot(
        plot_args=(x, y),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_fluxes(
    self,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    y = self.get_fluxes(normalise=normalise, concatenated=True)
    if y is None:
        return None, None
    return plot(
        plot_args=(y,),
        legend=y.columns,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_flux_selection(
    self,
    rate_names: list[str],
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    y = self.get_fluxes(normalise=normalise, concatenated=True)
    if y is None:
        return None, None
    y = cast(pd.DataFrame, y)
    return plot(
        plot_args=(y.loc[:, rate_names],),
        legend=rate_names,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_fluxes_grid(
    self,
    rate_groups: list[list[str]],
    *,
    ncols: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
    normalise: float | ArrayLike | None = None,
    plot_titles: Iterable[str] | None = None,
    figure_title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Array | None]:
    """Plot simulation results of the compound groups as a grid.

    Examples
    --------
    >>> plot_fluxes_grid([["v1", "v2"], ["v3", "v4]])

    """
    fluxes = self.get_fluxes(normalise=normalise, concatenated=True)
    if fluxes is None:
        return None, None
    fluxes = cast(pd.DataFrame, fluxes)
    plot_groups = [(cast(Array, fluxes.loc[:, group]),) for group in rate_groups]
    return plot_grid(
        plot_groups=plot_groups,  # type: ignore
        legend_groups=rate_groups,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        xlabels=xlabels,
        ylabels=ylabels,
        figure_title=figure_title,
        plot_titles=plot_titles,
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_fluxes_against_variable(
    self,
    variable: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if (c := self.get_results()) is None:
        return None, None
    if (v := self.get_fluxes(concatenated=True)) is None:
        return None, None
    rate_names = self.model.get_rate_names()
    x = c.loc[:, variable]
    y = v.loc[:, rate_names].to_numpy()
    return plot(
        plot_args=(x, y),
        legend=rate_names,
        xlabel=variable if xlabel is None else xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_flux_selection_against_variable(
    self,
    rate_names: list[str],
    variable: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if (c := self.get_results()) is None:
        return None, None
    if (v := self.get_fluxes()) is None:
        return None, None
    x = c.loc[:, variable]
    y = v.loc[:, rate_names].to_numpy()
    return plot(
        plot_args=(x, y),
        legend=rate_names,
        xlabel=variable if xlabel is None else xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_phase_plane(
    self,
    cpd1: str,
    cpd2: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    if (c := self.get_results()) is None:
        return None, None

    return plot(
        plot_args=(c.loc[:, cpd1], c.loc[:, cpd2]),
        legend=None,
        xlabel=cpd1 if xlabel is None else xlabel,
        ylabel=cpd2 if ylabel is None else ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_phase_space(
    self,
    cpd1: str,
    cpd2: str,
    cpd3: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    kwargs = _get_plot_kwargs(
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
        legend_kwargs=legend_kwargs,
    )
    kwargs["subplot"].update({"projection": "3d"})

    if (c := self.get_results()) is None:
        return None, None

    x = c.loc[:, cpd1]
    y = c.loc[:, cpd2]
    z = c.loc[:, cpd3]

    if x is None or y is None or z is None:
        return None, None

    xlabel = cpd1 if xlabel is None else xlabel
    ylabel = cpd2 if ylabel is None else ylabel
    zlabel = cpd3 if zlabel is None else zlabel

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"])
    else:
        fig = ax.get_figure()
    fig = cast(Figure, fig)
    ax = cast(Axis, ax)

    ax.plot(
        x,
        y,
        z,
        **kwargs["plot"],
    )
    _style_subplot(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        title=title,
        grid=grid,
        kwargs=kwargs,
    )
    if tight_layout:
        fig.tight_layout()
    return fig, ax


def plot_trajectories(
    self,
    cpd1: str,
    cpd2: str,
    *,
    cpd1_bounds: tuple[float, float],
    cpd2_bounds: tuple[float, float],
    n: int,
    y0: dict[str, float],
    t0: float = 0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    xlabel = cpd1 if xlabel is None else xlabel
    ylabel = cpd2 if ylabel is None else ylabel

    kwargs = _get_plot_kwargs(
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
        legend_kwargs=legend_kwargs,
    )

    x = np.linspace(*cpd1_bounds, n)
    y = np.linspace(*cpd2_bounds, n)
    u = np.zeros((n, n))
    v = np.zeros((n, n))

    fcd = self.model.get_full_concentration_dict(y=y0, t=t0)
    for i, s1 in enumerate(x):
        for j, s2 in enumerate(y):
            # Update y0 to new values
            fcd.update({cpd1: s1, cpd2: s2})
            rhs = self.model.get_right_hand_side(y=fcd, t=t0)
            u[i, j] = rhs[f"d{cpd1}dt"]
            v[i, j] = rhs[f"d{cpd2}dt"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"])
    else:
        fig = ax.get_figure()
    fig = cast(Figure, fig)
    ax = cast(Axis, ax)

    ax.quiver(x, y, u.T, v.T)
    _style_subplot(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        kwargs=kwargs,
    )
    if tight_layout:
        fig.tight_layout()
    return fig, ax


def plot_3d_trajectories(
    self,
    cpd1: str,
    cpd2: str,
    cpd3: str,
    *,
    cpd1_bounds: tuple[float, float],
    cpd2_bounds: tuple[float, float],
    cpd3_bounds: tuple[float, float],
    n: int,
    y0: dict[str, float],
    t0: float = 0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    kwargs = _get_plot_kwargs(
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
        legend_kwargs=legend_kwargs,
    )
    kwargs["subplot"].update({"projection": "3d"})

    x = np.linspace(*cpd1_bounds, n)
    y = np.linspace(*cpd2_bounds, n)
    z = np.linspace(*cpd3_bounds, n)
    u = np.zeros((n, n, n))
    v = np.zeros((n, n, n))
    w = np.zeros((n, n, n))

    fcd = self.model.get_full_concentration_dict(y=y0, t=t0)
    for i, s1 in enumerate(x):
        for j, s2 in enumerate(y):
            for k, s3 in enumerate(y):
                fcd.update({cpd1: s1, cpd2: s2, cpd3: s3})
                rhs = self.model.get_right_hand_side(y=fcd, t=t0)
                u[i, j, k] = rhs[f"d{cpd1}dt"]
                v[i, j, k] = rhs[f"d{cpd2}dt"]
                w[i, j, k] = rhs[f"d{cpd3}dt"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=kwargs["subplot"], **kwargs["figure"])
    else:
        fig = ax.get_figure()
    fig = cast(Figure, fig)
    ax = cast(Axis, ax)

    X, Y, Z = np.meshgrid(x, y, z)
    ax.quiver(
        X,
        Y,
        Z,
        np.transpose(u, [1, 0, 2]),
        np.transpose(v, [1, 0, 2]),
        np.transpose(w, [1, 0, 2]),
        length=0.05,
        normalize=True,
        alpha=0.5,
    )
    xlabel = cpd1 if xlabel is None else xlabel
    ylabel = cpd2 if ylabel is None else ylabel
    zlabel = cpd3 if zlabel is None else zlabel
    _style_subplot(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        title=title,
        grid=grid,
        kwargs=kwargs,
    )
    if tight_layout:
        fig.tight_layout()
    return fig, ax


def plot(
    self,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    """Plot all total concentrations."""
    compounds = sorted(
        [f"{i}__total" for i in self.model.label_compounds]
        + self.model.nonlabel_compounds
    )
    y = self.get_full_results(concatenated=True, include_readouts=False)
    if y is None:
        return None, None
    y = y.loc[:, compounds]
    return plot(
        plot_args=(y,),
        legend=None,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def _calculate_label_distribution(
    self, *, compound: str, relative: bool
) -> Array | None:
    """Calculate the label distribution of a compound."""
    total_concentration = self.get_total_concentration(compound=compound)
    if total_concentration is None:
        return None
    concentrations = []
    for position in range(
        self.model.get_compound_number_of_label_positions(compound=compound)
    ):
        concentration = self.get_concentration_at_positions(
            compound=compound, positions=position
        )
        if concentration is None:
            return None
        if relative:
            concentration = concentration / total_concentration
        concentrations.append(concentration)
    return np.array(concentrations).T


def plot_label_distribution(
    self,
    compound: str,
    relative: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    prefix: str = "Pos ",
    initial_index: int = 0,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    """Plot label distribution of a compound."""
    if ylabel is None and relative:
        ylabel = "Relative concentration"
    x = self.get_time()
    y = self._calculate_label_distribution(compound=compound, relative=relative)
    if x is None or y is None:
        return None, None

    return plot(
        plot_args=(x, y),
        legend=self._make_legend_labels(
            prefix=prefix, compound=compound, initial_index=initial_index
        ),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_label_distribution_grid(
    self,
    compounds: list[str],
    relative: bool = True,
    ncols: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    xlabels: str | list[str] | None = None,
    ylabels: str | list[str] | None = None,
    plot_titles: Iterable[str] | None = None,
    figure_title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    prefix: str = "Pos ",
    initial_index: int = 0,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Array | None]:
    """Plot label distributions of multiple compounds on a grid."""
    time = self.get_time()
    if time is None:
        return None, None
    plot_groups = [
        (
            time,
            self._calculate_label_distribution(compound=compound, relative=relative),
        )
        for compound in compounds
    ]
    legend_groups = [
        self._make_legend_labels(
            prefix=prefix, compound=compound, initial_index=initial_index
        )
        for compound in compounds
    ]
    if ylabels is None and relative:
        ylabels = "Relative concentration"
    if plot_titles is None:
        plot_titles = compounds
    return plot_grid(
        plot_groups=plot_groups,  # type: ignore
        legend_groups=legend_groups,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        xlabels=xlabels,
        ylabels=ylabels,
        figure_title=figure_title,
        plot_titles=plot_titles,
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_all_label_distributions(
    self,
    relative: bool = True,
    ncols: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    xlabels: str | list[str] | None = None,
    ylabels: str | list[str] | None = None,
    plot_titles: Iterable[str] | None = None,
    figure_title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    prefix: str = "Pos ",
    initial_index: int = 0,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Array | None]:
    """Plot label distributions of all compounds on a grid."""
    time = self.get_time()
    if time is None:
        return None, None
    compounds = self.model.label_compounds
    plot_groups = [
        (
            time,
            self._calculate_label_distribution(compound=compound, relative=relative),
        )
        for compound in compounds
    ]
    legend_groups = [
        self._make_legend_labels(
            prefix=prefix, compound=compound, initial_index=initial_index
        )
        for compound in compounds
    ]
    if ylabels is None and relative:
        ylabels = "Relative concentration"

    return plot_grid(
        plot_groups=plot_groups,  # type: ignore
        legend_groups=legend_groups,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        xlabels=xlabels,
        ylabels=ylabels,
        figure_title=figure_title,
        plot_titles=plot_titles,
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


# Linear label models


def plot_label_distribution(
    self,
    compound: str,
    relative: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    legend_prefix: str = "Pos ",
    initial_index: int = 0,
) -> tuple[Figure, Axis]:
    """Plot label distribution of a compound."""
    if ylabel is None and relative:
        ylabel = "Relative concentration"
    x = self.get_time()
    y = self.get_label_distribution(compound=compound)
    legend = self._make_legend_labels(legend_prefix, compound, initial_index)
    if title is None:
        title = compound
    return plot(
        plot_args=(x, y),
        legend=legend,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_label_distribution_grid(
    self,
    compounds: list[str],
    relative: bool = True,
    ncols: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    xlabels: str | list[str] | None = None,
    ylabels: str | list[str] | None = None,
    plot_titles: list[str] | None = None,
    figure_title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    legend_prefix: str = "Pos ",
    initial_index: int = 0,
) -> tuple[Figure | None, Array | None]:
    """Plot label distributions of multiple compounds on a grid."""
    time = self.get_time()
    plot_groups = [
        (time, self.get_label_distribution(compound=compound)) for compound in compounds
    ]
    legend_groups = [
        self._make_legend_labels(legend_prefix, compound, initial_index)
        for compound in compounds
    ]
    if ylabels is None and relative:
        ylabels = "Relative concentration"
    if plot_titles is None:
        plot_titles = compounds
    return plot_grid(
        plot_groups=plot_groups,  # type: ignore
        legend_groups=legend_groups,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        xlabels=xlabels,
        ylabels=ylabels,
        figure_title=figure_title,
        plot_titles=plot_titles,
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_all_label_distributions(
    self,
    relative: bool = True,
    ncols: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    xlabels: str | list[str] | None = None,
    ylabels: str | list[str] | None = None,
    plot_titles: Iterable[str] | None = None,
    figure_title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    legend_prefix: str = "Pos ",
    initial_index: int = 0,
) -> tuple[Figure | None, Array | None]:
    """Plot label distributions of all compounds on a grid."""
    time = self.get_time()
    compounds = self.model.isotopomers
    plot_groups = [
        (time, self.get_label_distribution(compound=compound)) for compound in compounds
    ]
    legend_groups = [
        self._make_legend_labels(legend_prefix, compound, initial_index)
        for compound in compounds
    ]
    if ylabels is None and relative:
        ylabels = "Relative concentration"
    if plot_titles is None:
        plot_titles = compounds

    return plot_grid(
        plot_groups=plot_groups,  # type: ignore
        legend_groups=legend_groups,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        xlabels=xlabels,
        ylabels=ylabels,
        figure_title=figure_title,
        plot_titles=plot_titles,
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


# Simulator


def plot(
    self,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalise: float | ArrayLike | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    ax: Axis | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any | None] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | None, Axis | None]:
    """Plot simulation results for a selection of compounds."""
    compounds = self.model.get_compounds()
    y = self.get_full_results(normalise=normalise, concatenated=True)
    if y is None:
        return None, None
    return plot(
        plot_args=(y.loc[:, compounds],),
        legend=compounds,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=grid,
        tight_layout=tight_layout,
        ax=ax,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )


def plot_producing_and_consuming(
    self,
    compound: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    figure_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_kwargs: dict[str, Any | None] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> tuple[None, None] | tuple[Figure, Axes]:
    producing: list[Array] = []
    consuming: list[Array] = []
    producing_names: list[str] = []
    consuming_names: list[str] = []

    if (v := self.get_fluxes()) is None:
        return None, None

    title = compound if title is None else title

    for rate_name, factor in self.model.stoichiometries_by_compounds[compound].items():
        if factor > 0:
            producing.append(v[rate_name].to_numpy() * factor)  # type: ignore
            producing_names.append(rate_name)
        else:
            consuming.append(v[rate_name].to_numpy() * -factor)  # type: ignore
            consuming_names.append(rate_name)

    return plot_grid(
        plot_groups=[
            (v.index.to_numpy(), np.array(producing).T),
            (v.index.to_numpy(), np.array(consuming).T),
        ],
        legend_groups=[producing_names, consuming_names],
        xlabels=xlabel,
        ylabels=ylabel,
        figure_title=title,
        plot_titles=("Producing", "Consuming"),
        grid=grid,
        tight_layout=tight_layout,
        figure_kwargs=figure_kwargs,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_kwargs=tick_kwargs,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
    )
