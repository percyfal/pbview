"""
DataStore class
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, override

import numpy as np
import pandas as pd
import pbzarr
import xarray as xr
import zarr

from pbview.logging import app_logger as logger
from pbview.model.coordinates import Coordinates
from pbview.model.track import Track

xr.set_options(display_expand_attrs=False)


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
        track = list(self.store.keys())[0]
        sample_sets = None
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

        self.base_coord = Coordinates.from_datatree(
            self.store[track], sample_sets=sample_sets
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

    def summary(self) -> None:
        return {
            "path": str(self.path),
            "tracks": self.tracks,
        }
