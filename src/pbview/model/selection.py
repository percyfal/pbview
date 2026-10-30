"""Selection state classes and helpers"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from functools import lru_cache

import numpy as np
import pandas as pd
import param

from pbview import config
from pbview.model.coordinates import ColumnSelection, Coordinates


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
        params.setdefault("coord", datastore.base_coord)
        super().__init__(**params)

        self.datastore = datastore

        # Populate selector objects
        self.param.samples.objects = list(datastore.samples)
        self.samples = list(datastore.samples)
        self.param.contigs.objects = list(datastore.contigs)
        self.contigs = list(datastore.contigs)
        self.param.missing_cutoff.objects = datastore.missing_cutoffs
        self.missing_cutoff = datastore.missing_cutoffs[0]
        tracks = list(datastore.tracks.keys())
        self.param.active_track.objects = tracks
        self.active_track = active_track if active_track in tracks else tracks[0]

    def _genome(self):
        return self.coord.datastore.genome  # SHIM: remove with Coordinates.datastore

    @param.depends("lower", "upper", "samples", "contigs", watch=True)
    def _recompute_coord(self):
        self.coord = self._filtered(
            self.coord.reset(),
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
        lower: int,
        upper: float,
        columns: tuple[ColumnSelection, ...],
        contigs: tuple[str, ...],
    ) -> Coordinates:
        genome = coords.datastore.genome  # SHIM REFACTOR(datastore): pass genome in
        keep = genome.contigs_in_length_range(lower, upper, contigs=contigs or None)
        for sel in columns:
            coords = coords.with_columns(sel)
        return coords.with_contigs(keep)

    def _current(self) -> Coordinates:
        return self._filtered(
            self.coord,
            lower=self.lower,
            upper=self.upper,
            columns=(ColumnSelection("sample", tuple(map(str, self.samples)) or None),),
            contigs=tuple(
                map(str, self.contigs),
            ),
        )

    @param.depends("lower", "upper", "contigs", watch=True)
    def _apply_contig_filters(self):
        """Push contig length param changes into the mutable coord object."""
        requested = tuple(self.contigs) or None
        genome = self.coord.datastore.genome  # SHIM: remove with Coordinates.datastore
        keep = genome.contigs_in_length_range(
            lower=self.lower, upper=self.upper, contigs=requested
        )
        self.coord.with_contigs(contigs=keep)

    @param.depends("samples", watch=True)
    def _apply_sample_filters(self):
        """Push sample param changes into the mutable coord object."""
        self.coord.with_samples(samples=self.samples or None)

    @param.depends("lower", "upper", "contigs")
    def contigs_df(self):
        regions = self._genome().contig_regions(self._current().spatial)
        return pd.DataFrame(
            {
                "contig": [c for c, _ in regions],
                "contig_len": np.array([e for _, _, e in regions], dtype=np.int64),
            }
        )

    @param.depends("samples", "contigs")
    def sample_sets_df(self):
        df = pd.DataFrame(
            {
                "sample set": self.coord._default_sample_set_membership,
                "Active": self.coord.sample_selected,
                "Total": self.coord._samples_all,
            }
        )
        if not np.all(self.coord._sample_set_membership == config.DEFAULT_SAMPLE_SET):
            df = pd.concat(
                df,
                pd.DataFrame(
                    {
                        "sample_sets": self.coord._sample_set_membership,
                        "Active": self.coord.sample_selected,
                        "Total": self.coord._samples_all,
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


def make_selection_state_class(
    base_coord: Coordinates, defaults: dict, max_coverage: int = 1000
) -> type[SelectionStateBase]:
    params = {}
    sample_sets = base_coord.sample_set_names
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
        params[f"max_missing_samples_{s}"] = param.Integer(
            default=int(base_coord.with_sample_sets([s]).n_samples / 2),
            bounds=(
                0,
                base_coord.with_sample_sets([s]).n_samples,
            ),
            doc=(
                "A site with >= max_missing_samples is inaccessible. "
                "The threshold applies at the sample set level"
            ),
        )

    return type("SelectionState", (SelectionStateBase,), params)


def build_selection_state(datastore, active_track, defaults):
    """SelectionState factory. Initialize plotting UI with sensible defaults"""
    cls = make_selection_state_class(base_coord=datastore.base_coord, defaults=defaults)
    return cls(datastore=datastore, active_track=active_track)
