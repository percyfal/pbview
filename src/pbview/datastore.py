"""
Datastore model
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"

import json
import re
from collections.abc import Iterable
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, override

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pbzarr
import pyranges1 as pr
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

from pbview import config
from pbview.logging import app_logger as logger

xr.set_options(display_expand_attrs=False)


_THRESHOLD_RE = re.compile(r"^(?P<base>.+)_missingness_threshold=(?P<t>\d+)$")


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


class Coordinates:
    """Immutable class representation of active sample/contig
    coordinates.

    Representation of samples, contigs and contig lengths. Provides
    convenience functions to subset coordinates and to keep track of
    active masks. Every filter operation returns a *new* `Coordinates`
    object. The derived quantities are cached per instance.

    The full coordinate set is saved in private variables, meaning a
    `Coordinates` object always has access to the full data view.

    Variable and function names with suffix `_all` refer to the full
    coordinate set, while those without suffix refer to the active
    subset.

    """

    def __init__(
        self,
        *,
        samples: npt.ArrayLike | xr.DataArray,
        contigs: npt.ArrayLike | xr.DataArray,
        offsets: npt.ArrayLike | xr.DataArray,
        sample_sets: npt.ArrayLike | None = None,
        sample_mask: npt.NDArray[np.bool_] | None = None,
        contig_mask: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        self._samples_all: npt.NDArray = np.asarray(samples)
        self._contigs_all: npt.NDArray = np.asarray(contigs)
        self._offsets: npt.NDArray[np.int64] = np.asarray(offsets, dtype=np.int64)
        # Masks: True = hidden. Default = keep everything
        n_samples_all, n_contigs_all = self._samples_all.size, self._contigs_all.size
        self.sample_mask: npt.NDArray[np.bool_] = (
            np.zeros(n_samples_all, dtype=bool)
            if sample_mask is None
            else np.asarray(sample_mask, dtype=bool)
        )
        self.contig_mask: npt.NDArray[np.bool_] = (
            np.zeros(n_contigs_all, dtype=bool)
            if contig_mask is None
            else np.asarray(contig_mask, dtype=bool)
        )
        self._default_sample_set_membership: npt.NDArray = np.repeat(
            config.DEFAULT_SAMPLE_SET, self._samples_all.size
        )
        self._sample_set_membership: npt.NDArray = np.asarray(
            self._default_sample_set_membership if sample_sets is None else sample_sets
        )
        user_sample_set_names = (
            [] if sample_sets is None else sorted(list(set(self.sample_set_membership)))
        )
        self.user_sample_set_names = np.asarray(user_sample_set_names)
        self.sample_set_names = np.asarray(
            [config.DEFAULT_SAMPLE_SET] + user_sample_set_names
        )
        # Freeze the underlying arrays to catch accidental mutation.
        for arr in (
            self._samples_all,
            self._contigs_all,
            self._offsets,
            self.sample_mask,
            self.contig_mask,
            self._default_sample_set_membership,
            self._sample_set_membership,
            self.user_sample_set_names,
            self.sample_set_names,
        ):
            arr.setflags(write=False)

    # FIXME: add n_user_sample_sets_all
    def __str__(self):
        return (
            f"Coordinates summary:\n"
            f"  Samples: {self.n_samples} ({self.n_samples_all})\n"
            f"  Contigs: {self.n_contigs} ({self.n_contigs_all})\n"
            f"  Sample sets: {self.n_user_sample_sets} "
            # f"({self.n_user_sample_sets_all})\n"
        )

    def __repr__(self) -> str:
        return (
            f"Coordinates(contigs={self.n_contigs}/{self.n_contigs_all}, "
            f"samples={self.n_samples}/{self.n_samples_all}, "
            f"sample_sets={self.n_user_sample_sets}/###, "
            # f"{self.n_user_sample_sets_all}, "
            f"genome_size={self.genome_size} bp)"
        )

    def __hash__(self) -> int:
        return hash(
            (
                id(self._samples_all),
                id(self._contigs_all),
                id(self._offsets),
                self.sample_mask.tobytes(),
                self.contig_mask.tobytes(),
            )
        )

    def _replace(
        self,
        *,
        sample_mask: npt.NDArray[np.bool_] | None = None,
        contig_mask: npt.NDArray[np.bool_] | None = None,
    ) -> "Coordinates":
        """Return a new instance sharing raw data, with new masks."""
        new = self.__class__.__new__(self.__class__)
        new._samples_all = self._samples_all
        new._contigs_all = self._contigs_all
        new._offsets = self._offsets
        new._default_sample_set_membership = self._default_sample_set_membership
        new._sample_set_membership = self._sample_set_membership
        new.sample_mask = (
            self.sample_mask
            if sample_mask is None
            else np.asarray(sample_mask, dtype=bool)
        )
        new.contig_mask = (
            self.contig_mask
            if contig_mask is None
            else np.asarray(contig_mask, dtype=bool)
        )
        new.sample_mask.setflags(write=False)
        new.contig_mask.setflags(write=False)
        user_sample_set_names = sorted(list(set(new.sample_set_membership)))
        new.user_sample_set_names = np.asarray(user_sample_set_names)
        new.sample_set_names = np.asarray(
            [config.DEFAULT_SAMPLE_SET] + user_sample_set_names
        )
        return new

    def matching_sample_set(self) -> str | None:
        """Return the name of the sample set that matches the current sample mask."""
        for name in self.sample_set_names:
            if name == "ALL":
                members = self._samples_all
            else:
                members = self._samples_all[self._sample_set_membership == name]
            if np.array_equal(np.sort(members), np.sort(self.samples)):
                return name
        return None

    # Factory methods
    def with_length_filter(
        self, lower: int = 0, upper: float = np.inf
    ) -> "Coordinates":
        """Return a new Coordinates whose contigs satisfy lower <= len <= upper."""
        keep = (
            (self._contig_len_all >= lower)
            & (self._contig_len_all <= upper)
            & ~self.contig_mask
        )
        return self._replace(contig_mask=~keep)

    def with_contigs(
        self,
        contigs: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed contigs. `None` resets the contig mask."""
        if contigs is None:
            return self._replace(contig_mask=np.zeros_like(self.contig_mask))
        keep = np.isin(self._contigs_all, np.asarray(list(contigs))) & ~self.contig_mask
        return self._replace(contig_mask=~keep)

    def with_samples(
        self,
        samples: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed samples. `None` resets the sample mask."""
        if samples is None:
            return self._replace(sample_mask=np.zeros_like(self.sample_mask))
        keep = np.isin(self._samples_all, np.asarray(list(samples))) & ~self.sample_mask
        return self._replace(sample_mask=~keep)

    def with_sample_sets(self, sample_sets: list[str]):
        """Keep only listed sample sets. `None` resets the sample mask."""
        if self.default_sample_set_name in sample_sets:
            return self
        if sample_sets is None:
            return self._replace(sample_mask=np.zeros_like(self.sample_mask))
        keep = np.isin(self._sample_set_membership, sample_sets) & ~self.sample_mask
        return self._replace(sample_mask=~keep)

    def reset(self, *, samples: bool = True, contigs: bool = True) -> "Coordinates":
        """Return a new Coordinates with masks reset (nothing hidden)."""
        return self._replace(
            sample_mask=(np.zeros_like(self.sample_mask) if samples else None),
            contig_mask=(np.zeros_like(self.contig_mask) if contigs else None),
        )

    @property
    def base_coord(self) -> "Coordinates":
        """Return a new Coordinates with no masks (nothing hidden)."""
        return self._replace(
            sample_mask=np.zeros_like(self.sample_mask),
            contig_mask=np.zeros_like(self.contig_mask),
        )

    @classmethod
    def from_datatree(
        cls, group: xr.DataTree, sample_sets: npt.ArrayLike | None = None
    ) -> "Coordinates":
        """Make base coordinates from DataTree"""
        return cls(
            samples=group.sample.values,
            contigs=group.contigs.values,
            offsets=group.offsets.values,
            sample_sets=sample_sets,
        )

    @cached_property
    def _contig_len_all(self) -> npt.NDArray[np.uint32]:
        return np.diff(self._offsets).astype(np.uint32)

    @property
    def contig_len_all(self):
        return self._contig_len_all

    @property
    def contig_len(self):
        return self._contig_len_all[~self.contig_mask]

    @property
    def contig_mask_is_active(self) -> bool:
        return bool(np.any(self.contig_mask))

    @property
    def contig_idx(self) -> dict:
        """Return a dict mapping of chromosome to index.
        Returns the mapping for the full set of contigs"""
        return {c: i for i, c in enumerate(self._contigs_all)}

    @cached_property
    def contig_indices(self) -> npt.NDArray[np.int64]:
        """Original-vector indices of currently selected contigs."""
        return np.flatnonzero(~self.contig_mask).astype(np.int64)

    def contig_slices(self) -> list[np.array[int]] | None:
        """Return a list of reduced slices mapped to position coordinates."""
        if self.contig_indices.size == 0:
            return []
        return (
            pr.PyRanges(
                {
                    "Start": self._offsets[:-1][self.contig_indices],
                    "End": self._offsets[1:][self.contig_indices],
                    "Chromosome": "Placeholder",
                }
            )
            .merge_overlaps(slack=1)[["Start", "End"]]
            .to_numpy()
            .tolist()
        )

    @property
    def contigs(self):
        return self._contigs_all[~self.contig_mask]

    @property
    def n_contigs(self):
        return self.contigs.size

    @property
    def n_contigs_all(self) -> int:
        return int(self._contigs_all.size)

    @property
    def samples(self) -> Any:
        return self._samples_all[~self.sample_mask]

    @property
    def n_samples(self):
        return int(self.samples.size)

    @property
    def n_samples_all(self):
        return int(self._samples_all.size)

    @property
    def sample_set_membership(self):
        """Return sample sets per sample"""
        return self._sample_set_membership[~self.sample_mask]

    @property
    def default_sample_set_membership(self):
        return (self._default_sample_set_membership[~self.sample_mask],)

    @property
    def n_sample_sets(self):
        """Return sample sets size, including the default set."""
        return self.sample_set_names.size

    @property
    def n_user_sample_sets(self):
        """Return user sample sets size"""
        return self.user_sample_set_names.size

    @property
    def default_sample_set_name(self) -> str:
        return self.sample_set_names[0]

    @property
    def has_sample_sets(self):
        """Return True if there are any sample sets other than the default set."""
        return self.n_user_sample_sets > 0

    @cached_property
    def genome_size(self):
        """Return the selected genome size. This is not a fixed entity"""
        return np.sum(self.contig_len, dtype=np.int64)

    @property
    def genome_size_all(self):
        """Return the total genome size, including masked contigs"""
        return np.sum(self.contig_len_all, dtype=np.int64)

    def as_pyranges(self):
        """Return pyranges object of active contigs"""
        return pr.PyRanges(
            {
                "Chromosome": self.contigs,
                "Start": np.zeros(self.n_contigs, dtype=np.int64),
                "End": self.contig_len,
            }
        )

    def sample_set_membership_dataframe(self, *, active_only=False) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "sample_set": self._default_sample_set_membership,
                "active": ~self.sample_mask,
                "sample": self._samples_all,
            }
        )
        if self.has_sample_sets:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "sample_set": self._sample_set_membership,
                            "active": ~self.sample_mask,
                            "sample": self._samples_all,
                        }
                    ),
                ]
            )
        if active_only:
            df = df[df.active]
        return df

    def sample_set_summary(self) -> pd.DataFrame:
        return (
            self.sample_set_membership_dataframe()
            .groupby("sample_set")
            .agg({"active": "sum", "sample": "count"})
            .rename({"active": "Active", "sample": "Total"}, axis=1)
        )

    def coordinates_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [self.n_contigs, self.n_contigs_all],
                [self.n_samples, self.n_samples_all],
            ],
            columns=["Active", "Total"],
            index=["Contigs", "Samples"],
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame summary of active samples, contigs and sample sets."""
        return pd.concat(
            [self.coordinates_dataframe(), self.sample_set_summary()],
            keys=["Coordinates", "Sample sets"],
            names=["Type", "Label"],
        )

    def _contigs_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "contig": self.contigs,
                "contig_len": self.contig_len,
            }
        )

    def summary(self) -> pd.DataFrame:
        """Return a data frame summary of coordinate selection relative to
        base coordinates."""
        return pd.DataFrame(
            [
                {
                    "genome_size_all": self.genome_size_all,
                    "genome_size": self.genome_size,
                    "genome_size (%)": np.round(
                        self.genome_size / self.genome_size_all * 100.0, 2
                    ),
                    "n_contigs": f"{self.n_contigs}/{self.n_contigs_all}",
                    "n_samples": f"{self.n_samples}/{self.n_samples_all}",
                    "n_sample_set_membership": (
                        f"{self.n_user_sample_sets}"  # /{self.n_user_sample_sets_all}"
                    ),
                }
            ]
        )


class PrecomputedTrack:
    """A representation of a precomputed data track."""

    def __init__(self, dt: xr.DataTree | None, track_name: str = "depth"):
        if dt is None:
            self.sum = None
            self.missingness = {}
            return

        self.sum = dt.get(f"{track_name}_sum")
        self.missingness = {}
        for key in dt:
            m = _THRESHOLD_RE.match(key)
            if m and m.group("base") == track_name:
                self.missingness[int(m.group("t"))] = dt[key]


class Track:
    """A representation of a pbzarr data track.

    Args:
        track_name: name of track
        data: pbzarr data group
        preprocess_data: preprocessing data set

    Returns:
        A Track object.
    """

    TARGET_BYTES = config.TARGET_BYTES

    def __init__(
        self,
        track_name: str,
        data: xr.DataTree,
        preprocess_data: xr.DataTree | None = None,
    ):
        self.name = track_name
        self._data = data
        self._pre = PrecomputedTrack(preprocess_data, track_name=track_name)

    def __repr__(self) -> str:
        return f"<Track(name={self.name})>"

    def __str__(self) -> str:
        return f"Track(name={self.name})"

    def _select_positions(self, base: xr.DataArray, coord: Coordinates) -> xr.DataArray:
        """Apply a contig/position selection from coord."""
        if not coord.contig_mask_is_active:
            return base
        if coord.n_contigs == 0:
            return base.sel(position=[])
        parts = [base.isel(position=slice(a, b)) for a, b in coord.contig_slices()]
        return xr.concat(parts, dim="position")

    def _select(self, base: xr.DataArray, coord: Coordinates) -> xr.DataArray:
        """Apply a sample + position selection"""
        res = base.sel(sample=coord.samples)
        return self._select_positions(res, coord)

    def data(self, coord: Coordinates) -> xr.DataArray:
        """Default view (original chunking)."""
        return self._select(self._data, coord)

    def _optimal_chunks(self, reduce_along: str) -> dict:
        """Return optimal chunk sizes for a dask array based on target bytes."""
        logger.debug(
            "Calculating optimal chunks for target bytes: %s", self.TARGET_BYTES
        )
        da = self._data["values"]
        itemsize = da.dtype.itemsize
        reduce_len = da.sizes[reduce_along]
        other_dim = next(d for d in da.dims if d != reduce_along)
        other_chunk = max(1, self.TARGET_BYTES // (reduce_len * itemsize))
        other_chunk = min(other_chunk, da.sizes[other_dim])
        return {
            reduce_along: -1,
            other_dim: int(other_chunk),
        }

    @property
    def _hist_view(self):
        """Rechunk data for histogram calculations.
        Ensure chunks are across all samples."""
        logger.debug("Rechunking data for histogram calculations")
        return self._data.chunk(self._optimal_chunks("sample"))

    @lru_cache(maxsize=64)
    def _coverage_hist_cached(self, bins: npt.NDArray[int], coord: Coordinates):
        logger.info("Calculating histogram of coverage for '%s' track", self.name)
        logger.debug("Current selection: %s", coord)
        bins = np.asarray(bins)
        name = coord.matching_sample_set()
        if name is not None and self._pre.sum is not None:
            sums = self._select_positions(self._pre.sum["values"], coord).sel(
                sample_set=name
            )

        else:
            # Raw path
            arr = self._select(self._hist_view, coord)["values"]
            sums = arr.sum("sample").astype(np.int32)
        max_bin = int(bins[-1])
        sums_clipped = da.minimum(sums.data, max_bin)
        hist = da.bincount(sums_clipped, minlength=int(bins[-1]) + 1)
        with ProgressBar():
            return hist.compute(), bins

    def coverage_hist(
        self,
        bins: npt.NDArray[int],
        coord: Coordinates,
    ):
        """Calculate coverage histogram for a given number of bins and coordinate state.

        The histogram values are calculated as the sum of values across all samples.

        Args:
            bins: number of bins for histogram
            coord: coordinate state for data selection
        Returns:
            Histogram, bins as a numpy arrays
        """
        return self._coverage_hist_cached(tuple(bins.tolist()), coord)

    @lru_cache(maxsize=64)
    def _missingness_hist_cached(
        self, bins: npt.NDArray, coord: Coordinates, threshold: int
    ):
        logger.info("Calculating missingness histogram for '%s' track", self.name)
        logger.debug("Current selection: %s", coord)
        bins = np.asarray(bins)
        name = coord.matching_sample_set()

        if name is not None and self._pre.missingness.get(threshold) is not None:
            counts = (
                self._select_positions(
                    self._pre.missingness[threshold]["values"], coord
                )
                .sel(sample_set=name)
                .data
            )
        else:
            # Raw path
            arr = self._select(self._hist_view, coord)["values"]
            counts = (arr > threshold).astype(np.int32).sum("sample").data
        max_bin = int(bins[-1])
        counts_clipped = da.minimum(counts, max_bin)
        with ProgressBar():
            hist = da.bincount(counts_clipped, minlength=int(bins[-1])).compute()
        return hist, bins

    def missingness_hist(
        self,
        coord: Coordinates,
        threshold: int = 0,
    ):
        """Calculate missingness histogram for a given number of bins
        and coordinate state.

        The histogram values are calculated as the number of values across all
        samples greater than threshold.

        Args:
            coord: coordinate state for data selection
            threshold: threshold value to treat values as missing (default: 0)
        Returns:
            Histogram, bins as a numpy arrays
        """
        bins = np.arange(0, coord.n_samples + 1)
        return self._missingness_hist_cached(tuple(bins.tolist()), coord, threshold)

    # FIXME: would it be possible to cheaply calculate the *combined*
    # effect of coverage and missing data filters? This would require
    # applying selected thresholds to *all* positions jointly and
    # returning the .dims["position"] attribute. Modifying the
    # thresholds for the individual histograms is cheap; the joint
    # operation is not. Would need reactive "UPDATE" button.
    def size(self):
        pass

    def summary(self, coord: Coordinates, lower: int = 0, upper: float = np.inf):
        """Return a track summary of coordinate selection relative to
        base coordinates."""
        df = pd.DataFrame(
            [
                {
                    "genome_size_all": coord.genome_size_all,
                    "genome_size": coord.genome_size,
                    "genome_size (%)": np.round(
                        coord.genome_size / coord.genome_size_all * 100.0, 2
                    ),
                    "min_contig_len": lower,
                    "max_contig_len": upper,
                }
            ]
        )
        return df


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
