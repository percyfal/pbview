"""
Track class and helpers
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import panel as pn

if TYPE_CHECKING:
    import numpy.typing as npt

    from pbview.model.coordinates import ColumnSelection
import xarray as xr
from dask.diagnostics import ProgressBar

from pbview import config
from pbview.logging import app_logger as logger
from pbview.model.coordinates import resolve_columns
from pbview.model.grouping import (
    GROUPING_NAME,
    IDENTITY_DIM,
    identity_grouping,
    validate_track_groupings,
)

from ._identity import datatree_id
from .chunking import optimal_chunks_dask

if TYPE_CHECKING:
    from pbview.model.coordinates import ColumnSelection, Coordinates, PositionIntervals


_THRESHOLD_RE = re.compile(r"^(?P<base>.+)_missing_cutoff=(?P<t>\d+)$")


def select_set(values: xr.DataArray, name: str) -> xr.DataArray:
    """Select one set of a summary track along its column dim.

    The column dim is `sample_set` for summaries of a user grouping and
    `identity` for summaries of the identity.
    """  # SHIM REFACTOR(precomputed): goes away with PrecomputedTrack
    (dim,) = [d for d in values.dims if d != "position"]
    return values.sel({dim: name})


class PrecomputedTrack:
    # FIXME(pbzarr>0.6): PrecomputedTrack will live in DataStore
    """A representation of a precomputed data track."""

    def __init__(self, dt: xr.DataTree | None, track_name: str = "depth"):
        self._local_cache = {}
        self._contig_mean_cache: dict[str, np.ndarray] = {}
        self._id = datatree_id(dt, schema_version=config.SCHEMA_VERSION)
        if dt is None:
            self.sum = None
            self.missing_cutoff = {}
            return

        self.sum = dt.get(f"{track_name}_sum")
        self.missing_cutoff = {}
        for key in dt:
            m = _THRESHOLD_RE.match(key)
            if m and m.group("base") == track_name:
                self.missing_cutoff[int(m.group("t"))] = dt[key]

    @pn.cache
    def _sum_hist_cached(
        self, dataset_id: str, sample_set: str, max_coverage_bin: int
    ) -> np.ndarray:
        sums = select_set(self.sum["values"], sample_set).data
        approx_max = int(sums.max().compute())
        if approx_max > max_coverage_bin:
            logger.warning(
                "Coverage sum histogram exceeds configured maximum (%d > %d). "
                "Truncating to max coverage bin.",
                approx_max,
                max_coverage_bin,
            )
            approx_max = max_coverage_bin
        return da.bincount(sums, minlength=approx_max + 1).compute()

    def sum_hist(self, sample_set: str, coord: Coordinates) -> np.ndarray:
        if sample_set not in self._local_cache:
            max_coverage_bin = min(
                coord.n_samples * config.MAX_COVERAGE_PER_SAMPLE,
                config.MAX_COVERAGE_BIN,
            )
            self._local_cache[sample_set] = self._sum_hist_cached(
                self._id, sample_set, max_coverage_bin
            )
        return self._local_cache[sample_set]

    def contig_mean_coverage(self, sample_set: str) -> np.ndarray:
        if sample_set not in self._contig_mean_cache:
            # FIXME: make progress optional
            with ProgressBar():
                self._contig_mean_cache[sample_set] = self._compute_contig_mean(
                    sample_set
                )
        return self._contig_mean_cache[sample_set]

    def _compute_contig_mean(self, sample_set: str) -> np.ndarray:
        logger.info("Calculating mean contig coverages")
        sums_da = select_set(self.sum["values"], sample_set).data
        offsets = self.sum["offsets"].values
        lengths = np.diff(offsets)
        n_contigs = len(lengths)

        def _partial_reduceat(chunk, block_info=None):
            start = block_info[0]["array-location"][0][0]
            end = block_info[0]["array-location"][0][1]
            first = np.searchsorted(offsets, start, side="right") - 1
            last = np.searchsorted(offsets, end, side="left")
            local_offsets = np.clip(offsets[first : last + 1] - start, 0, len(chunk))
            if len(local_offsets) < 2:
                return np.zeros(n_contigs, dtype=np.int64)
            reduced = np.add.reduceat(chunk, local_offsets[:-1])
            partial = np.zeros(n_contigs, dtype=np.int64)
            partial[first : first + len(reduced)] = reduced
            return partial.reshape(1, -1)

        per_chunk = sums_da.map_blocks(
            _partial_reduceat, dtype=np.int64, chunks=(1, n_contigs), new_axis=1
        )
        per_contig_sum = per_chunk.sum(axis=0).compute()
        return per_contig_sum / lengths


def _isel_intervals(data: xr.Dataset, intervals: PositionIntervals):
    """isel intervals from data along the position dimension."""
    # FIXME: Check if intervals correspond to entire genome
    if intervals.n_positions == data.position.size:
        return data
    parts = [data.isel(position=slice(s, e)) for s, e in intervals.intervals]
    if not parts:
        return data.isel(position=slice(0, 0))
    if len(parts) == 1:
        return parts[0]
    return xr.concat(
        parts, dim="position", data_vars="minimal", coords="minimal", compat="override"
    )


class Track:
    """A representation of a pbzarr data track.

    Args:
        track_name: name of track
        data: pbzarr data group
        preprocess_data: preprocessing data set
        meta: metadata for the track

    Returns:
        A Track object.
    """

    TARGET_BYTES = config.TARGET_BYTES
    SPATIAL_DIMS = config.SPATIAL_DIMS

    def __init__(
        self,
        track_name: str,
        data: xr.DataTree,
        preprocess_data: xr.DataTree
        | None = None,  # SHIM: remove with PrecomputedTrack
        *,
        groupings: tuple[xr.DataArray, ...] | None = None,
    ):
        self.name = track_name
        self._data = data
        self._pre = PrecomputedTrack(preprocess_data, track_name=track_name)
        column_dims = [dim for dim in data.dims if dim not in self.SPATIAL_DIMS]
        if len(column_dims) > 1:
            raise ValueError(
                f"Multiple column dimensions found in track {track_name}: {column_dims}"
            )
        self._column_dim = data.attrs.get(
            "perbase:column_dim", column_dims[0] if column_dims else "sample"
        )
        stored = validate_track_groupings(
            groupings,
            self.column_dim,
            self._data[self.column_dim].values,
            self._data.dims,
        )
        derived = (
            {}
            if self.column_dim == IDENTITY_DIM
            else {IDENTITY_DIM: identity_grouping(self.columns, self.column_dim)}
        )
        self._groupings = MappingProxyType({**derived, **stored})

    def __repr__(self) -> str:
        return f"<Track(name={self.name})>"

    def __str__(self) -> str:
        return f"Track(name={self.name})"

    @property
    def group_dims(self) -> tuple[str, ...]:
        return tuple(self._groupings)

    @classmethod
    def from_tree(
        cls, name: str, data: xr.DataTree, raw: xr.DataTree | None = None, **legacy
    ) -> "Track":
        node = raw.children.get(name)
        gnode = node.children.get("_meta", {}).get(GROUPING_NAME)
        groupings = tuple(gnode.dataset.data_vars.values()) if gnode is not None else ()
        return cls(name, data, groupings=groupings, **legacy)

    @property
    def groupings(self) -> Mapping[str, xr.DataArray]:
        return self._groupings

    def resolve_columns(self, selections: Iterable[ColumnSelection]) -> tuple[str, ...]:
        return resolve_columns(
            selections, self.column_dim, self.columns, self.groupings
        )

    @property
    def column_dim(self) -> str:
        """Return the name of the column dimension."""
        return self._column_dim

    @property
    def columns(self) -> list[str]:
        """Return the list of column names."""
        return self._data[self.column_dim].values.tolist()

    @property
    def values(self) -> xr.DataArray:
        """Track values, dims (position, <column_dim>)."""
        return self._data["values"]

    @property
    def coords(self):
        """Coordinates of the track data (contigs, offsets, columns)"""
        return self._data.coords

    @property
    def attrs(self) -> dict:
        """Attributes of the track group."""
        return self._data.attrs

    def _select_positions(self, res, coord: Coordinates) -> xr.DataArray:
        # SHIM: legacy bridge: remove with Track.data(coord)
        intervals = coord.datastore.genome.resolve(coord.spatial)
        if isinstance(res, xr.DataTree):
            res = res.to_dataset()
        return _isel_intervals(res, intervals)

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
        return optimal_chunks_dask(
            self._data["values"], reduce_along, self.TARGET_BYTES
        )

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
            sums = select_set(
                self._select_positions(self._pre.sum["values"], coord), name
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
        self, bins: npt.NDArray, coord: Coordinates, missing_cutoff: int
    ):
        logger.info("Calculating missingness histogram for '%s' track", self.name)
        logger.debug("Current selection: %s", coord)
        bins = np.asarray(bins)
        name = coord.matching_sample_set()

        if (
            name is not None
            and self._pre.missing_cutoff.get(missing_cutoff) is not None
        ):
            counts = select_set(
                self._select_positions(
                    self._pre.missing_cutoff[missing_cutoff]["values"], coord
                ),
                name,
            ).data
        else:
            # Raw path
            arr = self._select(self._hist_view, coord)["values"]
            counts = (arr <= missing_cutoff).astype(np.int32).sum("sample").data
        max_bin = int(bins[-1])
        counts_clipped = da.minimum(counts, max_bin)
        with ProgressBar():
            hist = da.bincount(counts_clipped, minlength=max_bin).compute()
        return hist, bins

    def missingness_hist(
        self,
        coord: Coordinates,
        missing_cutoff: int = 0,
    ):
        """Calculate missingness histogram for a given number of bins
        and coordinate state.

        The histogram values are calculated as the number of values across all
        samples greater than missing_cutoff.

        Args:
            coord: coordinate state for data selection
            missing_cutoff: cutoff value to treat values as missing (default: 0)
        Returns:
            Histogram, bins as a numpy arrays
        """
        bins = np.arange(0, coord.n_samples + 1)
        return self._missingness_hist_cached(
            tuple(bins.tolist()), coord, missing_cutoff
        )

    # FIXME: would it be possible to cheaply calculate the *combined*
    # effect of coverage and missing data filters? This would require
    # applying selected thresholds to *all* positions jointly and
    # returning the .dims["position"] attribute. Modifying the
    # thresholds for the individual histograms is cheap; the joint
    # operation is not. Would need reactive "UPDATE" button.
    def size(self):
        pass
