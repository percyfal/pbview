"""
Classes for viewing data
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-01"

from abc import abstractmethod
from functools import lru_cache
from typing import Any

import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import param
from panel.viewable import Viewer

from pbview import config
from pbview.datastore import Coordinates, Track

pn.extension("tabulator")


class SelectionStateBase(param.Parameterized):
    """Shared state object of current selections.

    Apply transformations and subset data in Coordinates and
    Datastore.
    """

    coord = param.ClassSelector(class_=Coordinates, precedence=-1)

    lower = param.Integer(default=0, bounds=(0, None), doc="Minimum contig length")
    upper = param.Number(
        default=float("inf"), doc="Maximum contig length (inf = no limit)"
    )
    lower_coverage = param.Integer(default=0, bounds=(0, None), doc="Minimum coverage")
    upper_coverage = param.Integer(
        default=100, doc="Maximum coverage", bounds=(0, None)
    )
    missingness = param.Integer(default=0, doc="Maximum missingness")
    samples = param.ListSelector(default=[], objects=[])
    contigs = param.ListSelector(default=[], objects=[])

    # Track-related parameters
    active_track = param.Selector(default=None, objects=[])

    def __init__(self, datastore, active_track="depth", **params):
        params.setdefault("coord", datastore.base_coord)
        super().__init__(**params)

        self.datastore = datastore

        # Populate selector objects
        self.param.samples.objects = list(datastore.base_coord._samples)
        self.samples = list(datastore.base_coord._samples)
        self.param.contigs.objects = list(datastore.base_coord._contigs)
        self.contigs = list(datastore.base_coord._contigs)
        self.param.missingness.bounds = (0, self.coord.samples_size)

        tracks = list(datastore.tracks.keys())
        self.param.active_track.objects = tracks
        self.active_track = active_track if active_track in tracks else tracks[0]

    @param.depends("lower", "upper", "samples", "contigs", watch=True)
    def _recompute_coord(self):
        self.coord = self._filtered(
            self.coord.base_coord,
            self.lower,
            self.upper,
            tuple(self.samples),
            tuple(self.contigs),
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _filtered(
        base: Coordinates,
        lower: int,
        upper: float,
        samples: tuple[str, ...],
        contigs: tuple[str, ...],
    ) -> Coordinates:
        return base.with_samples(samples=samples or None).with_contigs(
            contigs=contigs or None, lower=lower, upper=upper
        )

    @param.depends("lower", "upper", "contigs", watch=True)
    def _apply_contig_filters(self):
        """Push contig length param changes into the mutable coord object."""
        self.coord.with_contigs(
            lower=self.lower, upper=self.upper, contigs=self.contigs
        )

    @param.depends("samples", watch=True)
    def _apply_sample_filters(self):
        """Push sample param changes into the mutable coord object."""
        self.coord.with_samples(samples=self.samples or None)

    @param.depends("lower", "upper", "contigs")
    def contigs_df(self):
        return pd.DataFrame(
            {
                "contig": self.coord.contigs,
                "contig_len": self.coord.contig_len,
            }
        )

    @param.depends("samples", "contigs")
    def sample_sets_df(self):
        df = pd.DataFrame(
            {
                "sample set": self.coord._default_sample_sets,
                "Active": ~self.coord.sample_mask,
                "Total": self.coord._samples,
            }
        )
        if not np.all(self.coord._sample_sets == config.DEFAULT_SAMPLE_SET):
            df = pd.concat(
                df,
                pd.DataFrame(
                    {
                        "sample_sets": self.coord._sample_sets,
                        "Active": ~self.coord.sample_mask,
                        "Total": self.coord._samples,
                    }
                ),
            )
        return df.groupby("sample set").agg({"Active": "sum", "Total": "count"})

    @param.depends("lower", "upper", "samples", "contigs")
    def summary_df(self):
        return self.coord.to_dataframe()

    @param.depends("active_track")
    def track(self):
        return self.datastore.tracks[self.active_track]


def make_selection_state(
    base_coord: Coordinates, max_coverage: int = 1000
) -> type[SelectionStateBase]:
    params = {}
    for s in base_coord.sample_sets_unique:
        params[f"lower_coverage_{s}"] = param.Integer(default=0, bounds=(0, None))
        params[f"upper_coverage_{s}"] = param.Integer(
            default=max_coverage, bounds=(0, None)
        )
        params[f"missingness_{s}"] = param.Integer(
            default=0, bounds=(0, base_coord.with_samples(sample_sets=[s]).samples_size)
        )

    return type("SelectionState", (SelectionStateBase,), params)


class CoordinatesView(Viewer):
    """Coordinates view for datastore"""

    def __init__(self, state: SelectionStateBase, **params):
        super().__init__(**params)
        self.state = state
        self.lower_w = pn.widgets.IntInput.from_param(
            state.param.lower, name="Min contig length", sizing_mode="stretch_width"
        )
        self.upper_w = pn.widgets.FloatInput.from_param(
            state.param.upper, name="Max contig length", sizing_mode="stretch_width"
        )
        self.samples_w = pn.Column(
            pn.widgets.MultiChoice.from_param(state.param.samples, name="Mask samples"),
            height=150,
            scroll=True,
        )
        self.contigs_w = pn.Column(
            pn.widgets.MultiChoice.from_param(state.param.contigs, name="Mask contigs"),
            height=150,
            scroll=True,
        )

        self.size_pane = pn.bind(
            lambda *_: pn.Column(
                pn.indicators.LinearGauge(
                    label="Genome size",
                    value=self.state.coord.size,
                    bounds=(0, self.state.coord.genome_size),
                    horizontal=True,
                    format="{value} bp",
                    width=75,
                    height=int(config.SIDEBAR_WIDTH * 0.9),
                ),
                pn.indicators.LinearGauge(
                    value=np.round(
                        self.state.coord.size / self.state.coord.genome_size * 100, 2
                    ),
                    bounds=(0, 100),
                    horizontal=True,
                    format="{value} %",
                    width=50,
                    colors=["red"],
                    height=int(config.SIDEBAR_WIDTH * 0.9),
                ),
            ),
            state.param.lower,
            state.param.upper,
            state.param.samples,
            state.param.contigs,
        )

        self.summary_pane = pn.bind(
            lambda *_: pn.widgets.Tabulator(
                state.summary_df(), groupby=["Type"], hidden_columns=["Type"]
            ),
            state.param.lower,
            state.param.upper,
            state.param.samples,
            state.param.contigs,
        )
        self.hist_pane = pn.panel(
            pn.bind(
                self._hist, state.param.lower, state.param.upper, state.param.samples
            ),
            loading_indicator=True,
        )

    def _hist(self, *_):
        return self.state.contigs_df().hvplot.hist(
            y="contig_len",
            bins=20,
            width=int(config.SIDEBAR_WIDTH * 0.95),
        )

    def __panel__(self):
        return pn.Column(
            self.summary_pane,
            self.samples_w,
            self.contigs_w,
            pn.Column(
                "### Contig selection and statistics",
                self.lower_w,
                self.upper_w,
            ),
            self.hist_pane,
        )


class TrackView(Viewer, param.ParameterizedABC):
    """Base abstract class for track views"""

    def __init__(self, track: Track, state: SelectionStateBase, **params):
        super().__init__(**params)
        self.track = track
        self.state = state

    @abstractmethod
    def __panel__(self) -> Any:
        pass


class TrackSummaryView(TrackView):
    def __panel__(self):
        return pn.Column("# Track summary", self.track.summary(self.state.coord))


class _TrackPlotView(TrackView, param.ParameterizedABC):
    """Generic TrackPlotView class"""

    sample_set = param.String(
        default=config.DEFAULT_SAMPLE_SET, doc="Sample set to plot"
    )

    @abstractmethod
    def hist(self, *_):
        pass


class TrackCoverageView(_TrackPlotView):
    maxbins = param.Integer(default=1000, bounds=(0, None), doc="Maximum bin size")
    plot_type = param.Selector(default="area", objects=["area", "bar"], doc="Plot type")

    def __init__(self, **params):
        super().__init__(**params)
        self._df = None
        self.compute_accessible = pn.bind(
            self._compute_accessible,
            self.state.param[f"lower_coverage_{self.sample_set}"],
            self.state.param[f"upper_coverage_{self.sample_set}"],
        )

    @property
    def bins(self):
        return np.arange(self.maxbins + 2)

    def _data(self):
        if self.sample_set == config.DEFAULT_SAMPLE_SET:
            sample_sets = list(set(self.state.coord.sample_sets))
        else:
            sample_sets = [self.sample_set]
        counts, bins = self.track.coverage_hist(
            bins=self.bins, coord=self.state.coord.with_samples(sample_sets=sample_sets)
        )
        bins = bins[:-1]
        self._df = pd.DataFrame({"bins": bins, "counts": counts})

    @param.depends("maxbins", "plot_type")
    def hist(self, *_):
        self._data()
        # func = getattr(self.data.hvplot, "area")
        # kw = {
        #     "fill_alpha": 0.1,
        #     "legend": True,
        #     "by": "sampleset", # feature
        # }
        # dims = dict(kdims=["bins"], vdims=["counts"])
        return self._df.hvplot.scatter(x="bins", y="counts", shared_axes=False)

    @param.depends("maxbins")
    def _compute_accessible(self, lower_coverage, upper_coverage):
        self._data()
        counts = self._df["counts"]
        idx = np.searchsorted(
            np.arange(len(counts)), [lower_coverage, int(upper_coverage)]
        )
        return np.sum(counts[idx[0] : idx[1]])

    def __panel__(self):
        return pn.Column(
            "# Coverage",
            pn.Row(
                self.param.maxbins,
                self.state.param[f"lower_coverage_{self.sample_set}"],
                self.state.param[f"upper_coverage_{self.sample_set}"],
            ),
            self.compute_accessible,
            self.hist,
        )


class TrackMissingnessView(_TrackPlotView):
    missingness_threshold = param.Integer(
        default=3,
        bounds=(0, None),
        doc="Minimum coverage to consider an individual sample site as accessible",
    )

    def __init__(self, **params):
        super().__init__(**params)
        self._df = None
        self.maxbins = self.state.coord.samples_size
        self.compute_accessible = pn.bind(
            self._compute_accessible,
            self.state.param.missingness,
        )

    def _data(self):
        counts, bins = self.track.missingness_hist(
            bins=self.bins, coord=self.state.coord, threshold=self.missingness_threshold
        )
        bins = bins[:-1]
        sample_sets = np.repeat(config.DEFAULT_SAMPLE_SET, len(counts))
        df = pd.DataFrame({"bins": bins, "counts": counts, "sampleset": sample_sets})
        for sample_set in list(set(self.state.coord.sample_sets)):
            _counts, _bins = self.track.missingness_hist(
                bins=self.bins,
                coord=self.state.coord.with_samples(sample_sets=[sample_set]),
                threshold=self.missingness_threshold,
            )
            _bins = _bins[:-1]
            _sample_sets = np.repeat(sample_set, len(_counts))
            _df = pd.DataFrame(
                {"bins": _bins, "counts": _counts, "sampleset": _sample_sets}
            )
            df = pd.concat([df, _df], ignore_index=True)
        self._df = df

    @property
    def bins(self):
        return np.arange(self.maxbins + 2)

    def hist(self, *_):
        self._data()
        return self._df.hvplot.scatter(
            x="bins", y="counts", by="sampleset", shared_axes=False
        )

    def __panel__(self):
        return pn.Column(
            "# Missingness",
            pn.Row(
                self.param.missingness_threshold,
                self.state.param.missingness,
            ),
            self.compute_accessible,
            self.hist,
        )

    def _compute_accessible(self, missingness):
        self._data()
        print("Missingness: ", missingness)
        print(self.state.coord.sample_sets_unique)
        counts = self._df["counts"]
        return np.sum(counts[missingness:])


class DataStoreView(Viewer):
    """Class representing the main data store.

    Load data from cache and respond to filters to create views of the
    data.
    """

    def __init__(self, datastore, active_track="depth", **params):
        super().__init__(**params)
        SelectionState = make_selection_state(base_coord=datastore.base_coord)
        self.state = SelectionState(datastore=datastore, active_track=active_track)
        self.title = datastore.title
        self.cv = CoordinatesView(state=self.state)
        self.track_coverage_view = {
            s: TrackCoverageView(
                track=self.state.track(), state=self.state, sample_set=s
            )
            for s in self.state.coord.sample_sets_unique
        }
        self.track_missingness_view = TrackMissingnessView(
            track=self.state.track(), state=self.state
        )

        self.main_pane = pn.bind(
            self._render_main,
            self.state.param.active_track,
            self.state.param.lower,
            self.state.param.upper,
            self.state.param.samples,
            self.state.param.contigs,
        )

    # FIXME: if the trackplotview is too expensive to recompute on
    # every coordinate selection one could bind to a separate function
    # where an active user input is required to refresh.
    def _render_main(self, *_):
        return pn.Column(
            pn.Row(
                pn.GridBox(
                    *[
                        self.track_coverage_view[s]
                        for s in self.state.coord.sample_sets_unique
                    ],
                    ncols=2,
                ),
                self.track_missingness_view,
            )
        )

    # FIXME: need a refresh button to reset to defaults. Possibly add
    # individual reset buttons to each selector?
    def sidebar(self) -> pn.Card:
        return pn.Column(
            pn.Card(
                pn.widgets.Select(
                    name="Active track",
                    options=list(self.state.param.active_track.objects),
                    value=self.state.param.active_track.default,
                    sizing_mode="stretch_width",
                ),
                title="Datastore View",
                collapsed=False,
                header_background=config.SIDEBAR_BACKGROUND,
                active_header_background=config.SIDEBAR_BACKGROUND,
                styles=config.VCARD_STYLE,
            ),
            pn.Card(
                self.cv,
                title="Coordinates summary",
                collapsed=False,
                header_background=config.SIDEBAR_BACKGROUND,
                active_header_background=config.SIDEBAR_BACKGROUND,
                styles=config.VCARD_STYLE,
                sizing_mode="stretch_width",
            ),
        )

    def __panel__(self):
        return pn.Column(self.main_pane)
