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


class SelectionState(param.Parameterized):
    """Shared state object of current selections.

    Apply transformations and subset data in Coordinates and
    Datastore.
    """

    base_coord = param.ClassSelector(class_=Coordinates, precedence=-1)
    coord = param.ClassSelector(class_=Coordinates, precedence=-1)

    lower = param.Integer(default=0, bounds=(0, None), doc="Minimum contig length")
    upper = param.Number(
        default=float("inf"), doc="Maximum contig length (inf = no limit)"
    )
    samples = param.ListSelector(default=[], objects=[])
    contigs = param.ListSelector(default=[], objects=[])

    # Track-related parameters
    active_track = param.Selector(default=None, objects=[])

    def __init__(self, datastore, active_track="depth", **params):
        params.setdefault("base_coord", datastore.base_coord)
        params.setdefault("coord", datastore.base_coord)
        super().__init__(**params)

        self.datastore = datastore

        # Populate selector objects
        self.param.samples.objects = list(datastore.base_coord._samples)
        self.samples = list(datastore.base_coord._samples)
        self.param.contigs.objects = list(datastore.base_coord._contigs)
        self.contigs = list(datastore.base_coord._contigs)

        tracks = list(datastore.tracks.keys())
        self.param.active_track.objects = tracks
        self.active_track = active_track if active_track in tracks else tracks[0]

    @param.depends("lower", "upper", "samples", "contigs", watch=True)
    def _recompute_coord(self):
        self.coord = self._filtered(
            self.base_coord,
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

    @param.depends("lower", "upper")
    def contigs_df(self):
        return pd.DataFrame(
            {
                "contig": self.coord.contigs,
                "contig_len": self.coord.contig_len,
            }
        )

    @param.depends("samples")
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


class CoordinatesView(Viewer):
    """Coordinates view for datastore"""

    def __init__(self, state: SelectionState, **params):
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
            height=200,
            scroll=True,
        )
        # self.contigs_w =  pn.widgets.MultiSelect.from_param(state.param.contigs,
        #                                                     name="Toggle contigs")

        self.summary_pane = pn.bind(
            lambda *_: pn.widgets.Tabulator(
                state.summary_df(), groupby=["Type"], hidden_columns=["Type"]
            ),
            state.param.lower,
            state.param.upper,
            state.param.samples,
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
            # self.contigs_w,
            pn.Column(
                "### Contig selection and statistics",
                self.lower_w,
                self.upper_w,
            ),
            self.hist_pane,
        )


class TrackView(Viewer, param.ParameterizedABC):
    """Base abstract class for track views"""

    def __init__(self, track: Track, state: SelectionState, **params):
        super().__init__(**params)
        self.track = track
        self.state = state

    @abstractmethod
    def __panel__(self) -> Any:
        pass


class TrackSummaryView(TrackView):
    def __panel__(self):
        return pn.Column("# Track summary", self.track.summary(self.state.coord))


class DataStoreView(Viewer):
    """Class representing the main data store.

    Load data from cache and respond to filters to create views of the
    data.
    """

    def __init__(self, datastore, active_track="depth", **params):
        super().__init__(**params)
        self.state = SelectionState(datastore=datastore, active_track=active_track)
        self.title = datastore.title
        self.cv = CoordinatesView(state=self.state)

        self.main_pane = pn.bind(
            self._render_main,
            self.state.param.active_track,
            self.state.param.lower,
            self.state.param.upper,
            self.state.param.samples,
        )

    def _render_main(self, *_):
        track = self.state.track()
        tsv = TrackSummaryView(state=self.state, track=track)
        return pn.Column(tsv)

    def sidebar(self) -> pn.Card:
        return pn.Column(
            pn.Card(
                pn.Column(
                    self.state.param.active_track,
                    width=int(config.SIDEBAR_WIDTH * 0.95),
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
