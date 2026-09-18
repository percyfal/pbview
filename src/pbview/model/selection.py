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
from pbview.model.coordinates import Coordinates


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
        self.param.samples.objects = list(datastore.base_coord._samples_all)
        self.samples = list(datastore.base_coord._samples_all)
        self.param.contigs.objects = list(datastore.base_coord._contigs_all)
        self.contigs = list(datastore.base_coord._contigs_all)
        self.param.missingness.bounds = (0, self.coord.n_samples_all)

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
        return (
            base.with_samples(samples=samples or None)
            .with_contigs(contigs=contigs or None)
            .with_length_filter(lower=lower, upper=upper)
        )

    @param.depends("lower", "upper", "contigs", watch=True)
    def _apply_contig_filters(self):
        """Push contig length param changes into the mutable coord object."""
        self.coord.with_contigs(contigs=self.contigs).with_length_filter(
            lower=self.lower, upper=self.upper
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
                "sample set": self.coord._default_sample_set_membership,
                "Active": ~self.coord.sample_mask,
                "Total": self.coord._samples_all,
            }
        )
        if not np.all(self.coord._sample_set_membership == config.DEFAULT_SAMPLE_SET):
            df = pd.concat(
                df,
                pd.DataFrame(
                    {
                        "sample_sets": self.coord._sample_set_membership,
                        "Active": ~self.coord.sample_mask,
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
        params[key] = param.Integer(default=defaults[key], bounds=(0, None))
        key = f"upper_coverage_{s}"
        params[key] = param.Integer(default=defaults[key], bounds=(0, None))
        params[f"missingness_{s}"] = param.Integer(
            default=int(base_coord.with_sample_sets([s]).n_samples / 2),
            bounds=(
                0,
                base_coord.with_sample_sets([s]).n_samples,
            ),
        )

    return type("SelectionState", (SelectionStateBase,), params)


def build_selection_state(datastore, active_track, defaults):
    """SelectionState factory. Initialize plotting UI with sensible defaults"""
    cls = make_selection_state_class(base_coord=datastore.base_coord, defaults=defaults)
    return cls(datastore=datastore, active_track=active_track)
