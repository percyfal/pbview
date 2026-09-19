"""
DataStore class
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import functools
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
        self.path = Path(path)
        try:
            self.store = pbzarr.open(self.path)
        except zarr.errors.GroupNotFoundError as e:
            logger.error("Error opening pbzarr store: %s", e)
            raise
        except pbzarr.PbzError as e:
            logger.error("Error opening pbzarr store: %s", e)
            raise
        # FIXME: The coordinates will later on live in the root group
        track = list(self.store.keys())[0]
        # FIXME: temporary solution to retrieving the coordinates
        self._store_coords = self.store[track].coords
        sample_sets = None
        self.sample_set_names = [config.DEFAULT_SAMPLE_SET]
        if sampleinfo is not None:
            sampleinfo_df = pd.read_table(
                sampleinfo,
                header=None,
                sep=r"\s+",
                index_col=0,
                names=["sample", "sample_set"],
            )
            samples = self.store[track].coords["sample"].values
            sample_sets = sampleinfo_df.loc[samples]["sample_set"].values
            self.sample_set_names.extend(list(set(sample_sets)))

        self.base_coord = Coordinates(
            self,
            sample_sets=sample_sets,
        )
        self.tracks = {
            name: Track(
                track_name=name,
                data=self.store[name],
                preprocess_data=xr.open_datatree(path, engine="zarr", chunks={}),
            )
            for name in self.store.keys()
        }

    def __repr__(self) -> str:
        return f"<DataStore(path={self.path}, tracks={self.tracks})>"

    def __str__(self) -> str:
        return f"DataStore(path={self.path}, datasets={self.tracks})"

    @property
    def title(self):
        return self.path

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
    def default_sample_set_membership(self) -> npt.NDArray:
        return np.repeat(config.DEFAULT_SAMPLE_SET, self.n_samples)

    @functools.cached_property
    def sample_set_colors(self) -> dict[str, str]:
        import colorcet as cc

        if len(self.sample_set_names) > 8:
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

    def summary(self) -> None:
        return {
            "path": str(self.path),
            "tracks": self.tracks,
        }
