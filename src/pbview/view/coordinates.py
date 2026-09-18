"""
Classes for viewing Coordinates
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"


import hvplot.pandas  # noqa
import numpy as np
import panel as pn
from panel.viewable import Viewer

from pbview import config
from pbview.model.selection import SelectionStateBase

pn.extension("tabulator")


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

        self.genome_size_pane = pn.bind(
            lambda *_: pn.Column(
                pn.indicators.LinearGauge(
                    label="Genome size",
                    value=self.state.coord.genome_size,
                    bounds=(0, self.state.coord.genome_size_all),
                    horizontal=True,
                    format="{value} bp",
                    width=120,
                    height=int(config.SIDEBAR_WIDTH * 0.9),
                ),
                pn.indicators.LinearGauge(
                    value=np.round(
                        self.state.coord.genome_size
                        / self.state.coord.genome_size_all
                        * 100,
                        2,
                    ),
                    bounds=(0, 100),
                    horizontal=True,
                    format="{value} %",
                    width=90,
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

    def _hist(self, *_):
        return self.state.contigs_df().hvplot.hist(
            y="contig_len",
            bins=20,
            width=int(config.SIDEBAR_WIDTH * 0.95),
        )

    def __panel__(self):
        return pn.Column(
            self.genome_size_pane,
            self.summary_pane,
            pn.Column(
                "### Contig selection on length",
                self.lower_w,
                self.upper_w,
            ),
        )
