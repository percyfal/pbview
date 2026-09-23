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
import pandas as pd
import pbzarr
import xarray as xr
import zarr

from pbview import config
from pbview.logging import app_logger as logger
from pbview.model.coordinates import Coordinates
from pbview.model.track import Track

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


class DataStore:
    def __init__(self, path: Path | str, sampleinfo: Path | str | None = None):
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
        self._active_track = list(self.store.keys())[0]
        # FIXME: temporary solution to retrieving the coordinates
        self._store_coords = self.store[self._active_track].coords

        self._has_sampleinfo = sampleinfo is not None
        self._user_sample_set_membership = self._parse_sampleinfo(sampleinfo)
        if self._has_sampleinfo:
            self._user_sample_set_membership.setflags(write=False)

        self.base_coord = Coordinates(self)
        self.tracks = {
            name: Track(
                track_name=name,
                data=self.store[name],
                preprocess_data=xr.open_datatree(path, engine="zarr", chunks={}),
            )
            for name in self.store.keys()
        }

    def __repr__(self) -> str:
        return f"<DataStore(path={self._path}, tracks={self.tracks}, id={self.id[:8]})>"

    def __str__(self) -> str:
        return f"DataStore(path={self._path}, datasets={self.tracks}, id={self.id[:8]})"

    def _parse_sampleinfo(self, sampleinfo) -> npt.NDArray | None:
        if sampleinfo is None:
            return None
        df = pd.read_table(
            sampleinfo,
            header=None,
            sep=r"\s+",
            index_col=0,
            names=["sample", "sample_set"],
        )
        missing = set(self.samples) - set(df.index)
        if missing:
            raise ValueError(f"Samples missing from sampleinfo: {sorted(missing)[:5]}")
        return df.loc[self.samples, "sample_set"].to_numpy(dtype=str)

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
                self._sampleinfo_hash(),  # None or hash of sampleinfo content
            ]
        ).encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _sampleinfo_hash(self) -> str:
        if not self._has_sampleinfo:
            return "no-sampleinfo"
        return hashlib.blake2b(
            self._user_sample_set_membership.tobytes(),
            digest_size=8,
        ).hexdigest()

    @property
    def title(self):
        return self._path

    @property
    def data(self) -> str:
        return self.store

    @functools.cached_property
    def samples(self) -> npt.NDArray:
        """All samples in the dataset"""
        return np.asarray(self._store_coords["sample"].values)

    @functools.cached_property
    def n_samples(self) -> int:
        return self.samples.size

    @functools.cached_property
    def contigs(self) -> npt.NDArray:
        """All contigs in the dataset"""
        return np.asarray(self._store_coords["contigs"].values)

    @functools.cached_property
    def n_contigs(self) -> int:
        return self.contigs.size

    @functools.cached_property
    def offsets(self) -> npt.NDArray:
        """Contig offsets"""
        return np.asarray(self._store_coords["offsets"].values)

    @functools.cached_property
    def default_sample_set_name(self) -> str:
        if self.has_sample_sets and len(set(self.sample_set_membership)) == 1:
            return str(self.sample_set_membership[0])
        return config.DEFAULT_SAMPLE_SET

    @functools.cached_property
    def default_sample_set_membership(self) -> npt.NDArray:
        return np.repeat(self.default_sample_set_name, self.n_samples)

    @functools.cached_property
    def sample_set_membership(self) -> npt.NDArray:
        if self._has_sampleinfo:
            return self._user_sample_set_membership
        return self.default_sample_set_membership

    @functools.cached_property
    def sample_set_names(self) -> list:
        return [self.default_sample_set_name, *self.user_sample_set_names]

    @functools.cached_property
    def n_sample_sets(self) -> int:
        return len(self.sample_set_names)

    @functools.cached_property
    def user_sample_set_names(self) -> list[str]:
        if not self.has_sample_sets:
            return []
        names = np.unique(self.sample_set_membership).tolist()
        if len(names) == 1:
            return []
        return sorted(names)

    @functools.cached_property
    def n_user_sample_set_names(self) -> int:
        return len(self.user_sample_set_names)

    @property
    def has_sample_sets(self) -> bool:
        return self._has_sampleinfo

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

    @functools.cached_property
    def missing_cutoffs(self) -> list[int]:
        # FIXME(pbzarr>0.6): PrecomputedTracks will live in DataStore
        return sorted(self._pre.missing_cutoff.keys())

    @property
    def _pre(self):
        # FIXME(pbzarr>0.6): remove this shim; use DataStore attribute
        return self.tracks[self._active_track]._pre

    def summary(self) -> None:
        return {
            "path": str(self._path),
            "tracks": self.tracks,
        }
