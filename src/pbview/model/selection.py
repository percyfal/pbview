"""Selection state classes and helpers"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import param

from pbview.model.coordinates import ColumnSelection, Coordinates, Genome
from pbview.model.thresholds import default_max_missing_samples

if TYPE_CHECKING:
    from pbview.model.track import Track


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
    samples = param.ListSelector(default=[], objects=[])
    contigs = param.ListSelector(default=[], objects=[])
    missing_cutoff = param.Selector(
        default=None,
        doc="An sample site is considered missing if the coverage <= this value",
    )

    # Track-related parameters
    active_track = param.Selector(default=None, objects=[])

    def __init__(self, datastore, active_track="depth", **params):
        params.setdefault("coord", Coordinates())
        super().__init__(**params)

        self.datastore = datastore

        # Populate selector objects
        self.param.samples.objects = list(datastore.samples)
        self.samples = list(datastore.samples)
        self.param.contigs.objects = list(datastore.contigs)
        self.contigs = list(datastore.contigs)
        if not datastore.missing_cutoffs:
            raise ValueError(
                "No missingness summaries found in the store; run `pbview preprocess`"
            )
        self.param.missing_cutoff.objects = datastore.missing_cutoffs
        self.missing_cutoff = datastore.missing_cutoffs[0]
        tracks = list(datastore.tracks.keys())
        self.param.active_track.objects = tracks
        self.active_track = active_track if active_track in tracks else tracks[0]

    @param.depends("lower", "upper", "samples", "contigs", watch=True)
    def _recompute_coord(self):
        self.coord = self._filtered(
            Coordinates(),
            self.datastore.genome,
            self.lower,
            self.upper,
            columns=(ColumnSelection("sample", tuple(map(str, self.samples)) or None),),
            contigs=tuple(
                map(str, self.contigs),
            ),
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _filtered(
        coords: Coordinates,
        genome: Genome,
        lower: int,
        upper: float,
        columns: tuple[ColumnSelection, ...],
        contigs: tuple[str, ...],
    ) -> Coordinates:
        keep = genome.contigs_in_length_range(lower, upper, contigs=contigs or None)
        for sel in columns:
            coords = coords.with_columns(sel)
        return coords.with_contigs(keep)

    def _current(self) -> Coordinates:
        return self._filtered(
            self.coord,
            self.datastore.genome,
            lower=self.lower,
            upper=self.upper,
            columns=(ColumnSelection("sample", tuple(map(str, self.samples)) or None),),
            contigs=tuple(
                map(str, self.contigs),
            ),
        )

    @param.depends("lower", "upper", "contigs")
    def contigs_df(self):
        regions = self.datastore.genome.contig_regions(self._current().spatial)
        return pd.DataFrame(
            {
                "contig": [c for c, _ in regions],
                "contig_len": np.array([e for _, _, e in regions], dtype=np.int64),
            }
        )

    @param.depends("active_track")
    def track(self):
        return self.datastore.tracks[self.active_track]


def make_selection_state_class(
    track: Track, base_coord: Coordinates, defaults: dict, max_coverage: int = 1000
) -> type[SelectionStateBase]:
    params = {}
    sample_sets = track.groups("sample_set", base_coord.column)
    for s in sample_sets:
        key = f"lower_coverage_{s}"
        params[key] = param.Integer(
            default=defaults[key],
            bounds=(0, None),
            label=f"Lower total coverage {s}",
            doc="Lower total coverage cutoff",
        )
        key = f"upper_coverage_{s}"
        params[key] = param.Integer(
            default=defaults[key],
            bounds=(0, None),
            label=f"Upper total coverage {s}",
            doc="Upper total coverage cutoff",
        )
        n_samples = track.n_selected(base_coord.with_sample_sets([s]).column)
        params[f"max_missing_samples_{s}"] = param.Integer(
            default=default_max_missing_samples(n_samples),
            bounds=(0, n_samples),
            doc=(
                "A site with >= max_missing_samples is inaccessible. "
                "The threshold applies at the sample set level"
            ),
        )

    return type("SelectionState", (SelectionStateBase,), params)


def build_selection_state(datastore, active_track, defaults):
    """SelectionState factory. Initialize plotting UI with sensible defaults"""
    track = datastore.tracks[active_track]
    cls = make_selection_state_class(track, Coordinates(), defaults)
    return cls(datastore=datastore, active_track=active_track)
