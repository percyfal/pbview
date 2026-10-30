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
import pandas as pd
from panel.viewable import Viewer
from typing import TYPE_CHECKING

from pbview import config
from pbview.model.selection import SelectionStateBase

if TYPE_CHECKING:
    from pbview.model.coordinates import Coordinates, Genome
    from pbview.model.track import Track


pn.extension("tabulator")


def selection_overview(
    coords: Coordinates, genome: Genome, track: Track
) -> pd.DataFrame:
    """Active and total number of contigs, columns and sample sets.

    The frame has a (Type, Label) index, as consumed by the sidebar table.
    """
    label = track.column_dim.replace("_", " ").capitalize() + "s"
    rows = [
        (
            "Coordinates",
            "Contigs",
            len(coords.selected_contigs(genome)),
            len(genome.contigs),
        ),
        ("Coordinates", label, track.n_selected(coords.column), len(track.columns)),
        *(
            ("Sample sets", name, active, total)
            for name, (active, total) in track.group_counts(
                "sample_set", coords.column
            ).items()
        ),
    ]
    return pd.DataFrame(rows, columns=["Type", "Label", "Active", "Total"]).set_index(
        ["Type", "Label"]
    )


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

        # FIXME: currently shows total size for all contigs
        genome = self.state.datastore.genome
        self.genome_size_pane = pn.bind(
            lambda *_: pn.Column(
                pn.indicators.LinearGauge(
                    label="Genome size",
                    value=genome.size,
                    bounds=(0, genome.size),
                    horizontal=True,
                    format="{value} bp",
                    width=120,
                    height=int(config.SIDEBAR_WIDTH * 0.9),
                ),
                pn.indicators.LinearGauge(
                    value=np.round(
                        genome.size / genome.size * 100,
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
                selection_overview(state.coord, state.datastore.genome, state.track()),
                groupby=["Type"],
                hidden_columns=["Type"],
            ),
            state.param.coord,
            state.param.active_track,
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
