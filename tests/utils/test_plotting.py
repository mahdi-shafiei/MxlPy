# type: ignore

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import pandas as pd

from modelbase2.utils.plotting import (
    _get_plot_kwargs,
    _style_subplot,
    heatmap_from_dataframe,
    plot,
    plot_grid,
)


class PlotTests(unittest.TestCase):
    def test_get_plot_kwargs(self):
        self.assertEqual(
            _get_plot_kwargs(
                figure_kwargs={},
                subplot_kwargs={},
                plot_kwargs={},
                grid_kwargs={},
                tick_kwargs={},
                label_kwargs={},
                title_kwargs={},
                legend_kwargs={},
            ),
            {
                "figure": {"figsize": (10, 7)},
                "subplot": {"facecolor": "white"},
                "plot": {"linewidth": 4},
                "grid": {
                    "color": (0, 0, 0),
                    "alpha": 0.33,
                    "linestyle": "dashed",
                    "linewidth": 1,
                },
                "ticks": {
                    "direction": "out",
                    "length": 6,
                    "width": 2,
                    "labelsize": 14,
                    "color": "0.15",
                    "pad": 7,
                },
                "label": {"fontsize": 14},
                "title": {"fontsize": 18},
                "legend": {
                    "loc": "upper left",
                    "bbox_to_anchor": (1.02, 1),
                    "borderaxespad": 0,
                    "ncol": 1,
                    "fontsize": 12,
                    "numpoints": 1,
                    "scatterpoints": 1,
                    "markerscale": 1,
                    "frameon": False,
                },
            },
        )

    def test_style_subplot(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="test")
        ax.set_title("Test")
        kwargs = _get_plot_kwargs(
            figure_kwargs={},
            subplot_kwargs={},
            plot_kwargs={},
            grid_kwargs={},
            tick_kwargs={},
            label_kwargs={},
            title_kwargs={},
            legend_kwargs={},
        )

        _style_subplot(ax=ax, kwargs=kwargs, legend=True, title=True)
        plt.close()

    def test_plot(self):
        fig, ax = plot(plot_args=([1, 2, 3], [1, 2, 3]), legend="legend", title="Title")
        plt.close()

    def test_plot_grid_without_legend(self):
        fig, axs = plot_grid([([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [2, 3, 4])])
        plt.close()

    def test_plot_grid_xlabels_str(self):
        fig, axs = plot_grid(
            [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [2, 3, 4])],
            sharex=False,
            sharey=False,
            xlabels="Test",
        )
        self.assertEqual(axs[0, 0].get_xlabel(), "Test")
        self.assertEqual(axs[0, 1].get_xlabel(), "Test")
        plt.close()

    def test_plot_grid_xlabels_iterable(self):
        fig, axs = plot_grid(
            [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [2, 3, 4])],
            sharex=False,
            sharey=False,
            xlabels=["Test1", "Test2"],
        )
        self.assertEqual(axs[0, 0].get_xlabel(), "Test1")
        self.assertEqual(axs[0, 1].get_xlabel(), "Test2")
        plt.close()

    def test_plot_grid_ylabels_str(self):
        fig, axs = plot_grid(
            [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [2, 3, 4])],
            sharex=False,
            sharey=False,
            ylabels="Test",
        )
        self.assertEqual(axs[0, 0].get_ylabel(), "Test")
        self.assertEqual(axs[0, 1].get_ylabel(), "Test")
        plt.close()

    def test_plot_grid_ylabels_iterable(self):
        fig, axs = plot_grid(
            [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [2, 3, 4])],
            sharex=False,
            sharey=False,
            ylabels=["Test1", "Test2"],
        )
        self.assertEqual(axs[0, 0].get_ylabel(), "Test1")
        self.assertEqual(axs[0, 1].get_ylabel(), "Test2")
        plt.close()


class HeatmapTests(unittest.TestCase):
    def test_heatmap_from_dataframe(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fig, ax, hm = heatmap_from_dataframe(df)
        plt.close()

    def test_heatmap_from_dataframe_symlog(self):
        df = pd.DataFrame([[-1001, -0.01], [0.01, 1001]])
        fig, ax, hm = heatmap_from_dataframe(df)

    def test_heatmap_from_dataframe_log(self):
        df = pd.DataFrame([[1001, 2], [3, 4]])
        fig, ax, hm = heatmap_from_dataframe(df)

    def test_heatmap_from_dataframe_existing_axes(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fig, ax = plt.subplots()
        fig, ax, hm = heatmap_from_dataframe(df, ax=ax)
        plt.close()
