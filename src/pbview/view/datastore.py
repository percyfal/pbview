"""
Classes for viewing DataStore
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"

import hvplot.pandas  # noqa
import panel as pn
from panel.viewable import Viewer
import param

from pbview import config
from pbview.model.histogram import compute_threshold_defaults
from pbview.model.selection import build_selection_state
from pbview.view.coordinates import CoordinatesView
from pbview.view.track import TrackCoveragePage, TrackMissingnessPage
from pbview.view.summary import SummaryPage

pn.extension("tabulator")


class DataStoreView(Viewer):
    """Class representing the main data store.

    Load data from cache and respond to filters to create views of the
    data.
    """

    active_page = param.Selector()

    def __init__(self, datastore, active_track="depth", **params):
        super().__init__(**params)
        defaults = compute_threshold_defaults(datastore, active_track)
        self.state = build_selection_state(datastore, active_track, defaults)
        self.title = str(datastore.title)
        self.cv = CoordinatesView(state=self.state)

        # Instantiate pages
        self.pages = {
            "Coverage": TrackCoveragePage(
                track=self.state.track(),
                state=self.state,
                maxbins_default={
                    s: defaults[f"maxbins_{s}"]
                    for s in self.state.coord.sample_set_names
                },
            ),
            "Missingness": TrackMissingnessPage(
                track=self.state.track(), state=self.state
            ),
        }
        self.pages["Summary"] = SummaryPage(
            coverage_page=self.pages["Coverage"],
            missingness_page=self.pages["Missingness"],
        )
        self.pages = {k: self.pages[k] for k in ["Summary", "Coverage", "Missingness"]}

        self.param.active_page.objects = list(self.pages)
        self.active_page = next(iter(self.pages))

    @param.depends("active_page")
    def main(self):
        return self.pages[self.active_page]

    def chooser(self):
        return pn.widgets.RadioButtonGroup.from_param(
            self.param.active_page, button_type="success"
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
