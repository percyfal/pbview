"""
Track class and helpers
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import re
from functools import lru_cache
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import panel as pn

if TYPE_CHECKING:
    import numpy.typing as npt
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from pbview import config
from pbview.logging import app_logger as logger
from pbview.model._identity import datatree_id

if TYPE_CHECKING:
    from pbview.model.coordinates import Coordinates


_THRESHOLD_RE = re.compile(r"^(?P<base>.+)_missingness_threshold=(?P<t>\d+)$")


class PrecomputedTrack:
    """A representation of a precomputed data track."""

    def __init__(self, dt: xr.DataTree | None, track_name: str = "depth"):
        self._local_cache = {}
        self._contig_mean_cache: dict[str, np.ndarray] = {}
        self._id = datatree_id(dt, schema_version=config.SCHEMA_VERSION)
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

    @pn.cache
    def _sum_hist_cached(self, dataset_id: str, sample_set: str) -> np.ndarray:
        sums = self.sum["values"].sel(sample_set=sample_set).data
        approx_max = int(sums.max().compute())
        return da.bincount(sums, minlength=approx_max + 1).compute()

    def sum_hist(self, sample_set: str) -> np.ndarray:
        if sample_set not in self._local_cache:
            self._local_cache[sample_set] = self._sum_hist_cached(self._id, sample_set)
        return self._local_cache[sample_set]

    def contig_mean_coverage(self, sample_set: str) -> np.ndarray:
        if sample_set not in self._contig_mean_cache:
            with ProgressBar():
                self._contig_mean_cache[sample_set] = self._compute_contig_mean(
                    sample_set
                )
        return self._contig_mean_cache[sample_set]

    def _compute_contig_mean(self, sample_set: str) -> np.ndarray:
        logger.info("Calculating mean contig coverages")
        sums = self.sum["values"].sel(sample_set=sample_set).values
        offsets = self.sum["offsets"].values
        lengths = np.diff(offsets)
        per_contig_sum = np.add.reduceat(sums, offsets[:-1])
        return per_contig_sum / lengths


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
            counts = (arr <= threshold).astype(np.int32).sum("sample").data
        max_bin = int(bins[-1])
        counts_clipped = da.minimum(counts, max_bin)
        with ProgressBar():
            hist = da.bincount(counts_clipped, minlength=max_bin).compute()
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
