"""
Classes for viewing Track
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"

from abc import abstractmethod
from typing import Any

import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models.widgets.tables import NumberFormatter
from panel.viewable import Viewer

from pbview import config
from pbview.model.histogram import hist_stats
from pbview.model.selection import SelectionStateBase
from pbview.model.track import Track

from .plots import hist_boxplot

pn.extension("tabulator")


class TrackIndicatorTableBase(Viewer):
    state = param.ClassSelector(class_=SelectionStateBase, is_instance=True)
    hist_rxs = param.Dict()  # {sample_set: ((counts, bins))}
    formatters = {}

    def __init__(self, **params):
        super().__init__(**params)
        deps = (
            list(self.hist_rxs.values()) + [self.state.param.coord] + self._extra_deps
        )
        self._df_rx = pn.bind(self._build_df, *deps)

    @property
    def _extra_deps(self, **params):
        retval = []
        return retval

    def __panel__(self):
        return pn.widgets.Tabulator(
            self._df_rx,
            disabled=True,
            layout="fit_data_table",
            sizing_mode="stretch_width",
            formatters=self.formatters,
        )


class TrackIndicatorCoverageTable(TrackIndicatorTableBase):
    right = {"text_align": "right"}
    formatters = {
        "lower_by_sample": NumberFormatter(format="0.0", **right),
        "upper_by_sample": NumberFormatter(format="0.0", **right),
        "frac_active_genome": NumberFormatter(format="0.0%", **right),
        "frac_full_genome": NumberFormatter(format="0.0%", **right),
        "mean": NumberFormatter(format="0.0", **right),
        "60%": NumberFormatter(format="0.0", **right),
        "70%": NumberFormatter(format="0.0", **right),
        "80%": NumberFormatter(format="0.0", **right),
        "90%": NumberFormatter(format="0.0", **right),
        "med + 1std": NumberFormatter(format="0.0", **right),
        "med + 2std": NumberFormatter(format="0.0", **right),
        "std": NumberFormatter(format="0.0", **right),
    }

    def __init__(self, page, **params):
        self._page = page
        super().__init__(**params)

    @property
    def _extra_deps(self, **params):
        retval = []
        for s in self.hist_rxs:
            retval.append(self.state.param[f"lower_coverage_{s}"])
            retval.append(self.state.param[f"upper_coverage_{s}"])
        retval.append(self._page.param.by_sample)
        return retval

    def _build_df(self, *args):
        n = len(self.hist_rxs)
        hists = args[:n]
        coord = args[n]
        thresh = args[n + 1 : -1]
        by_sample = args[-1]

        rows = []
        for i, s in enumerate(self.hist_rxs):
            n = coord.with_sample_sets([s]).n_samples if by_sample else 1
            counts, bins = hists[i]
            lo, up = thresh[2 * i], thresh[2 * i + 1]
            stats = hist_stats(counts, bins)
            mask = (bins >= lo / n) & (bins <= up / n)
            accessible = int(counts[mask].sum())
            rows.append(
                {
                    "sample_set": s,
                    "lower": lo,
                    "lower_by_sample": lo / coord.with_sample_sets([s]).n_samples,
                    "upper": up,
                    "upper_by_sample": up / coord.with_sample_sets([s]).n_samples,
                    "accessible_bp": accessible,
                    "frac_active_genome": accessible / coord.genome_size,
                    "frac_full_genome": accessible / coord.genome_size_all,
                    "median": stats["median"],
                    "mean": stats["mean"],
                    "60%": stats["mean"] * 0.6,
                    "70%": stats["mean"] * 0.7,
                    "80%": stats["mean"] * 0.8,
                    "90%": stats["mean"] * 0.9,
                    "med + 1std": stats["median"] + stats["std"],
                    "med + 2std": stats["median"] + 2 * stats["std"],
                    "std": stats["std"],
                }
            )
        return pd.DataFrame(rows)


class TrackIndicatorMissingnessTable(TrackIndicatorTableBase):
    right = {"text_align": "right"}
    formatters = {
        "max_missing_samples_by_sample": NumberFormatter(format="0.0%", **right),
        "frac_active_genome": NumberFormatter(format="0.0%", **right),
        "frac_full_genome": NumberFormatter(format="0.0%", **right),
    }

    @property
    def _extra_deps(self, **params):
        retval = []
        for s in self.hist_rxs:
            retval.append(self.state.param[f"max_missing_samples_{s}"])
        return retval

    def _build_df(self, *args):
        n = len(self.hist_rxs)
        hists = args[:n]
        coord = args[n]
        miss = args[n + 1 :]

        rows = []
        for i, s in enumerate(self.hist_rxs):
            counts, bins = hists[i]
            max_missing_samples = miss[i]
            mask = bins <= max_missing_samples
            accessible = int(counts[mask].sum())
            rows.append(
                {
                    "sample_set": s,
                    "max_missing_samples": max_missing_samples,
                    "max_missing_samples_by_sample": max_missing_samples
                    / coord.with_sample_sets([s]).n_samples,
                    "accessible_bp": accessible,
                    "frac_active_genome": accessible / coord.genome_size,
                    "frac_full_genome": accessible / coord.genome_size_all,
                }
            )
        return pd.DataFrame(rows)


class TrackView(Viewer, param.ParameterizedABC):
    """Base abstract class for track views"""

    state = param.ClassSelector(class_=SelectionStateBase)

    def __init__(self, track: Track, **params):
        super().__init__(**params)
        self.track = track

    @abstractmethod
    def __panel__(self) -> Any:
        pass


class _TrackPlotView(TrackView, param.ParameterizedABC):
    """Generic TrackPlotView class"""

    maxbins = param.Integer(default=1000, bounds=(0, None), doc="Maximum bin size")
    sample_set = param.String(
        default=config.DEFAULT_SAMPLE_SET, doc="Sample set to plot"
    )
    by_sample = param.Boolean(default=False)

    @property
    def coord(self):
        return self.state.coord.with_sample_sets(sample_sets=[self.sample_set])


class TrackCoverageView(_TrackPlotView):
    def __init__(self, page, hist_rx, **params):
        super().__init__(**params)
        self._page = page
        self._hist_rx = hist_rx
        lo = self.state.param[f"lower_coverage_{self.sample_set}"]
        hi = self.state.param[f"upper_coverage_{self.sample_set}"]
        self.plot = pn.bind(self._plot, self._hist_rx, lo, hi, page.param.by_sample)

    def _plot(self, hist_data, lower, upper, by_sample):
        n = self.coord.n_samples if by_sample else 1
        counts, bins = hist_data
        df = pd.DataFrame({"coverage": bins, "count": counts})
        scatter = df.hvplot.scatter(x="coverage", y="count").opts(shared_axes=False)
        band = hv.VSpan(lower / n, upper / n).opts(color="grey", alpha=0.2)
        return (scatter * band).opts(shared_axes=False)

    def __panel__(self):
        return pn.Column(
            pn.Row(
                self.state.param[f"lower_coverage_{self.sample_set}"],
                self.state.param[f"upper_coverage_{self.sample_set}"],
            ),
            self.plot,
        )


class TrackMissingnessView(_TrackPlotView):
    def __init__(self, hist_rx, **params):
        super().__init__(**params)
        self._hist_rx = hist_rx
        max_missing_samples = self.state.param[f"max_missing_samples_{self.sample_set}"]
        self.maxbins = self.state.coord.n_samples

        self.plot = pn.bind(self._plot, self._hist_rx, max_missing_samples)

    def _plot(self, hist_data, max_missing_samples):
        counts, bins = hist_data
        df = pd.DataFrame({"missingness": bins, "count": counts})
        scatter = df.hvplot.scatter(x="missingness", y="count").opts(shared_axes=False)
        band = hv.VSpan(0, max_missing_samples).opts(color="grey", alpha=0.2)
        return (scatter * band).opts(shared_axes=False)

    def __panel__(self):
        return pn.Column(
            pn.Row(self.state.param[f"max_missing_samples_{self.sample_set}"]),
            self.plot,
        )


# FIXME: add annotation-based views
class TrackCoveragePage(Viewer):
    """Track coverage summary page.

    Container and viewer for multiple TrackCoverageView instances.
    """

    track = param.ClassSelector(class_=Track, is_instance=True)
    state = param.ClassSelector(class_=SelectionStateBase, is_instance=True)
    active_sets = param.ListSelector(default=[], objects=[])  # populated in __init__
    by_sample = param.Boolean(default=False, doc="Normalize coverages by sample size")
    maxbins = param.Integer(default=100)
    maxbins_default = param.Dict(default={})

    def __init__(self, **params):
        super().__init__(**params)
        self.param.active_sets.objects = self.state.coord.sample_set_names
        self.active_sets = list(self.state.coord.sample_set_names)

        self._hist_rx = {
            s: pn.rx(self._compute_hist)(
                self.maxbins_default[s], self.state.param.coord, s, self.param.by_sample
            )
            for s in self.state.coord.sample_set_names
        }
        self._boxplot = pn.bind(
            self._make_boxplot,
            *self._hist_rx.values(),
        )
        self._boxplot_names = list(self._hist_rx.keys())

        self._plots = {
            s: TrackCoverageView(
                page=self,
                track=self.track,
                state=self.state,
                sample_set=s,
                hist_rx=self._hist_rx[s],
                maxbins=self.maxbins,
                by_sample=self.by_sample,
            )
            for s in self.state.coord.sample_set_names
        }

        self._table = TrackIndicatorCoverageTable(
            page=self,
            state=self.state,
            hist_rxs=self._hist_rx,  # dict of rx
        )

    def __panel__(self):
        # active-set widget drives the plot grid; table shows all sets always
        chooser = pn.widgets.CheckBoxGroup.from_param(
            self.param.active_sets, inline=True
        )
        plot_grid = pn.bind(self._render_plots, self.param.active_sets)
        return pn.Column(
            pn.Row(chooser, self.param.by_sample), self._table, self._boxplot, plot_grid
        )

    def _compute_hist(self, maxbins, coord, sample_set, by_sample):
        bins = np.arange(maxbins + 1)
        counts, bins = self.track.coverage_hist(
            bins=bins, coord=coord.with_sample_sets([sample_set])
        )
        if by_sample:
            bins = bins / coord.with_sample_sets([sample_set]).n_samples
        return counts, bins

    def _render_plots(self, active):
        return pn.GridBox(*[self._plots[s] for s in active], ncols=2)

    def _make_boxplot(self, *hist_tuples):
        hists = {}
        values = {}
        for name, (counts, bins) in zip(self._boxplot_names, hist_tuples):
            hists[name] = counts
            values[name] = bins
        color_map = self.state.datastore.sample_set_colors
        return hist_boxplot(
            hists,
            values,
            color_map=color_map,
            responsive=True,
            height=400,
            title="Coverage distribution",
        )

    @property
    def table(self):
        return self._table


class TrackMissingnessPage(Viewer):
    track = param.ClassSelector(class_=Track, is_instance=True)
    state = param.ClassSelector(class_=SelectionStateBase, is_instance=True)
    active_sets = param.ListSelector(default=[], objects=[])  # populated in __init__

    def __init__(self, **params):
        super().__init__(**params)
        self.param.active_sets.objects = self.state.coord.sample_set_names
        self.active_sets = list(self.state.coord.sample_set_names)
        self.maxbins = self.state.coord.n_samples
        self._hist_rx = {
            s: pn.rx(self._compute_hist)(
                self.state.param.coord, s, self.state.param.missing_cutoff
            )
            for s in self.state.coord.sample_set_names
        }

        self._plots = {
            s: TrackMissingnessView(
                track=self.track,
                state=self.state,
                sample_set=s,
                hist_rx=self._hist_rx[s],
            )
            for s in self.state.coord.sample_set_names
        }

        self._table = TrackIndicatorMissingnessTable(
            state=self.state,
            hist_rxs=self._hist_rx,  # dict of rx
        )

    def _compute_hist(self, coord, sample_set, missing_cutoff):
        return self.track.missingness_hist(
            coord=coord.with_sample_sets([sample_set]), missing_cutoff=missing_cutoff
        )

    def _render_plots(self, active):
        return pn.GridBox(*[self._plots[s] for s in active], ncols=2)

    def __panel__(self):
        # active-set widget drives the plot grid; table shows all sets always
        chooser = pn.widgets.CheckBoxGroup.from_param(
            self.param.active_sets, inline=True
        )
        plot_grid = pn.bind(self._render_plots, self.param.active_sets)
        return pn.Column(
            pn.Row(chooser, self.state.param.missing_cutoff), self._table, plot_grid
        )

    @property
    def table(self):
        return self._table
