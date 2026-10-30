"""
DataStore class
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import functools
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, override

import numpy as np
import numpy.typing as npt
import pbzarr
import xarray as xr
import zarr

from pbview import config
from pbview.logging import app_logger as logger
from pbview.model.coordinates import Coordinates, Genome
from pbview.model.grouping import grouping_hash
from pbview.model.track import Summaries, Track, find_summaries

xr.set_options(display_expand_attrs=False)


def _to_hex(color):
    if isinstance(color, str):
        return color
    r, g, b = color[:3]
    if max(r, g, b) <= 1:
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


class NpEncoder(json.JSONEncoder):
    @override
    def default(self, obj) -> int | float | Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(json.JSONEncoder, self).default(obj)

    @override
    def iterencode(self, obj, _one_shot: bool = False) -> Iterable[str]:
        # Pre-process the object to replace inf/nan
        obj = self._sanitize(obj)
        return super().iterencode(obj, _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float) or isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return 1e308 if obj > 0 else -1e308  # large but valid JSON number
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return self._sanitize(obj.tolist())
        return obj


def _base_track_name(store) -> str:
    """First track, in sorted order, with a `sample` column dimension.

    SHIM REFACTOR(datastore): the store-level universe goes away with
    Coordinates.datastore.
    """
    names = sorted(store.keys())
    return next((n for n in names if "sample" in store[n].coords), names[0])


class DataStore:
    def __init__(self, path: Path):
        self._path = Path(path)
        try:
            self.store = pbzarr.open(self._path)
        except zarr.errors.GroupNotFoundError as e:
            logger.error("Error opening pbzarr store: %s", e)
            raise
        except pbzarr.PbzError as e:
            logger.error("Error opening pbzarr store: %s", e)
            raise
        # FIXME: The coordinates will later on live in the root group
        self._active_track = _base_track_name(self.store)
        # FIXME: temporary solution to retrieving the coordinates
        self._store_coords = self.store[self._active_track].coords

        self._genome = Genome(
            contigs=self.store[self._active_track].contigs.values,
            offsets=self.store[self._active_track].offsets.values,
        )

        names = sorted(self.store.keys())
        raw = xr.open_datatree(self._path, engine="zarr", chunks={})
        # Summary tracks are recognised by their attrs, never by name
        summary_tracks = {
            n: Track.from_tree(n, self.store[n], raw)
            for n in names
            if "perbase:source_track" in raw.children[n].attrs
        }
        summaries = find_summaries(summary_tracks.values())
        self.tracks = {
            n: summary_tracks[n]
            if n in summary_tracks
            else Track.from_tree(n, self.store[n], raw, summaries=summaries.get(n))
            for n in names
        }
        self.base_coord = Coordinates(self)

    def __repr__(self) -> str:
        return f"<DataStore(path={self._path}, tracks={self.tracks}, id={self.id[:8]})>"

    def __str__(self) -> str:
        return f"DataStore(path={self._path}, datasets={self.tracks}, id={self.id[:8]})"

    @functools.cached_property
    def id(self) -> str:
        p = Path(self._path).resolve()
        marker = p / ".zgroup" if (p / ".zgroup").exists() else p / "zarr.json"
        stat = marker.stat()

        payload = "|".join(
            [
                f"v{config.SCHEMA_VERSION}",
                str(p),
                str(stat.st_mtime_ns),
                self._grouping_hash(),
            ]
        ).encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _grouping_hash(self) -> str:
        grouping = self._sample_set_grouping
        return "no-sample-sets" if grouping is None else grouping_hash(grouping)

    @property
    def title(self):
        return self._path

    @property
    def data(self) -> str:
        return self.store

    @property
    def genome(self) -> str:
        return self._genome

    @functools.cached_property
    def samples(self) -> npt.NDArray:
        """All samples in the dataset"""
        # FIXME: temporary hack to avoid KeyError when the sample
        # dimension is not present in the coordinates
        try:
            return np.asarray(self._store_coords["sample"].values)
        except KeyError:
            logger.warning(
                "Sample dimension not found in coordinates; returning empty array."
            )
            return np.asarray([])

    @functools.cached_property
    def n_samples(self) -> int:
        return self.samples.size

    @functools.cached_property
    def contigs(self) -> npt.NDArray:
        """All contigs in the dataset"""
        return np.asarray(self._genome.contigs)

    @functools.cached_property
    def n_contigs(self) -> int:
        return self.contigs.size

    @functools.cached_property
    def offsets(self) -> npt.NDArray:
        """Contig offsets"""
        return np.asarray(self._genome.offsets)

    @functools.cached_property
    def _sample_set_grouping(self) -> xr.DataArray | None:
        """Store `sample_set` grouping of the base track, if any.

        SHIM REFACTOR(datastore): store-level view of the sample sets;
        goes away when Coordinates resolves sets through the tracks
        """
        return self.tracks[self._active_track].groupings.get("sample_set")

    @functools.cached_property
    def default_sample_set_name(self) -> str:
        return config.DEFAULT_GROUPING_SET

    @functools.cached_property
    def sample_set_names(self) -> list:
        return [self.default_sample_set_name, *self.user_sample_set_names]

    @functools.cached_property
    def n_sample_sets(self) -> int:
        return len(self.sample_set_names)

    @functools.cached_property
    def user_sample_set_names(self) -> list[str]:
        grouping = self._sample_set_grouping
        if grouping is None:
            return []
        return [
            s
            for s in grouping["sample_set"].values.tolist()
            if s != self.default_sample_set_name
        ]

    @functools.cached_property
    def n_user_sample_set_names(self) -> int:
        return len(self.user_sample_set_names)

    @property
    def has_sample_sets(self) -> bool:
        return self._sample_set_grouping is not None

    @functools.cached_property
    def sample_set_colors(self) -> dict[str, str]:
        if self.n_sample_sets > 8:
            import colorcet as cc

            palette = cc.glasbey_category10
            return {
                s: _to_hex(palette[i % len(palette)])
                for i, s in enumerate(self.sample_set_names)
            }
        # Okabe-Ito palette
        palette = [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ]
        return {
            s: palette[i % len(palette)] for i, s in enumerate(self.sample_set_names)
        }

    def summaries_of(self, track_name: str) -> Summaries:
        """Summary tracks computed from `track_name` (via `perbase:source_track`)."""
        return self.tracks[track_name].summaries

    @functools.cached_property
    def missing_cutoffs(self) -> list[int]:
        return sorted(self.summaries_of(self._active_track).missingness)

    def summary(self) -> None:
        return {
            "path": str(self._path),
            "tracks": self.tracks,
        }
