"""
Track class and helpers
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np

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

from .chunking import optimal_chunks_dask

if TYPE_CHECKING:
    from pbview.model.coordinates import (
        ColumnSelection,
        PositionIntervals,
        ResolvedSelection,
    )


class MissingSummaryError(ValueError):
    """A summary track that is needed is not in the store."""


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


def select_data(
    base: xr.DataTree | xr.Dataset, resolved: ResolvedSelection
) -> xr.Dataset:
    """Select the resolved columns and positions of a track's data."""
    res = base.sel({resolved.column_dim: list(resolved.labels)})
    if isinstance(res, xr.DataTree):
        res = res.to_dataset()
    return _isel_intervals(res, resolved.intervals)


@dataclass(frozen=True, eq=False)
class Summaries:
    """Summary tracks computed from a base track.

    Recognised by their attrs, never by name: `perbase:source_track`,
    `perbase:provenance` (`config.SUMMARY_SUM`, `config.SUMMARY_MISSINGNESS`)
    and, for missingness, `perbase:missing_cutoff`. Stores preprocessed
    before these attrs existed have no summaries until `pbview preprocess`
    is run again.
    """

    sum: Track | None = None
    missingness: Mapping[int, Track] = field(default_factory=dict)


@dataclass(frozen=True, eq=False, repr=False)
class Track:
    """A representation of a pbzarr data track.

    Immutable. Compares and hashes by identity, so instances can key
    `lru_cache`.

    Args:
        name: name of track
        data: pbzarr data group
        stored: stored user groupings of the track columns
        summaries: summary tracks computed from this track

    Returns:
        A Track object.
    """

    name: str
    data: xr.DataTree
    stored: tuple[xr.DataArray, ...] = field(default=(), kw_only=True)
    summaries: Summaries = field(default_factory=Summaries, kw_only=True)

    TARGET_BYTES = config.TARGET_BYTES
    SPATIAL_DIMS = config.SPATIAL_DIMS

    def __post_init__(self):
        column_dims = [dim for dim in self.data.dims if dim not in self.SPATIAL_DIMS]
        if len(column_dims) > 1:
            raise ValueError(
                f"Multiple column dimensions found in track {self.name}: {column_dims}"
            )
        column_dim = self.data.attrs.get(
            "perbase:column_dim", column_dims[0] if column_dims else "sample"
        )
        object.__setattr__(self, "_column_dim", column_dim)
        stored = validate_track_groupings(
            self.stored,
            column_dim,
            self.data[column_dim].values,
            self.data.dims,
        )
        derived = (
            {}
            if column_dim == IDENTITY_DIM
            else {IDENTITY_DIM: identity_grouping(self.columns, column_dim)}
        )
        object.__setattr__(self, "_groupings", MappingProxyType({**derived, **stored}))

    def __repr__(self) -> str:
        return f"<Track(name={self.name})>"

    def __str__(self) -> str:
        return f"Track(name={self.name})"

    @property
    def group_dims(self) -> tuple[str, ...]:
        return tuple(self._groupings)

    @classmethod
    def from_tree(
        cls, name: str, data: xr.DataTree, raw: xr.DataTree | None = None, **kwargs
    ) -> "Track":
        if raw is None:
            return cls(name, data, stored=None, **kwargs)
        node = raw.children.get(name)
        gnode = node.children.get("_meta", {}).get(GROUPING_NAME)
        groupings = tuple(gnode.dataset.data_vars.values()) if gnode is not None else ()
        return cls(name, data, stored=groupings, **kwargs)

    @property
    def groupings(self) -> Mapping[str, xr.DataArray]:
        return self._groupings

    def resolve_columns(self, selections: Iterable[ColumnSelection]) -> tuple[str, ...]:
        return resolve_columns(
            selections, self.column_dim, self.columns, self.groupings
        )

    def n_selected(self, selections: Iterable[ColumnSelection]) -> int:
        """Number of columns selected by `selections`."""
        return len(self.resolve_columns(selections))

    def _grouping_for(self, dim: str) -> xr.DataArray | None:
        grouping = self.groupings.get(dim)
        return grouping if grouping is not None else self.groupings.get(IDENTITY_DIM)

    def groups(
        self, dim: str = "sample_set", selections: Iterable[ColumnSelection] = ()
    ) -> tuple[str, ...]:
        """Groups of `dim` with a member among the columns selected by `selections`.

        Falls back to the identity grouping when the track has no
        grouping for `dim`. The default set is always listed first.
        """
        return self._groups_of(self.resolve_columns(selections), dim)

    def _groups_of(self, labels: Iterable[str], dim: str) -> tuple[str, ...]:
        default = config.DEFAULT_GROUPING_SET
        grouping = self._grouping_for(dim)
        if grouping is None:  # a summary of the identity: its columns are the set
            return (default,)
        group_dim = grouping.dims[1]
        selected = np.isin(grouping[self.column_dim].values, list(labels))
        present = grouping.values[selected].any(axis=0)
        names = grouping[group_dim].values.tolist()
        return (default, *(n for n, p in zip(names, present) if p and n != default))

    def matching_group(
        self, selections: Iterable[ColumnSelection], dim: str = "sample_set"
    ) -> str | None:
        """Name of the group of `dim` whose members equal the selected columns."""
        return self._matching_group_of(self.resolve_columns(selections), dim)

    def _matching_group_of(self, labels: Iterable[str], dim: str) -> str | None:
        selected = set(labels)
        grouping = self._grouping_for(dim)
        if grouping is None:
            return (
                config.DEFAULT_GROUPING_SET if selected == set(self.columns) else None
            )
        group_dim = grouping.dims[1]
        columns = grouping[self.column_dim].values
        for name in self._groups_of(selected, dim):
            members = columns[grouping.sel({group_dim: name}).values]
            if set(members.tolist()) == selected:
                return name
        return None

    def group_counts(
        self, dim: str = "sample_set", selections: Iterable[ColumnSelection] = ()
    ) -> dict[str, tuple[int, int]]:
        """(selected, total) number of columns for each group of `dim`.

        Falls back to the identity grouping when the track has no
        grouping for `dim`.
        """
        grouping = self._grouping_for(dim)
        selected = self.resolve_columns(selections)
        if grouping is None:  # a summary of the identity
            return {config.DEFAULT_GROUPING_SET: (len(selected), len(self.columns))}
        group_dim = grouping.dims[1]
        chosen = np.isin(grouping[self.column_dim].values, selected)
        member = grouping.values
        return {
            name: (int(member[chosen, i].sum()), int(member[:, i].sum()))
            for i, name in enumerate(grouping[group_dim].values.tolist())
        }

    @property
    def column_dim(self) -> str:
        """Return the name of the column dimension."""
        return self._column_dim

    @property
    def columns(self) -> list[str]:
        """Return the list of column names."""
        return self.data[self.column_dim].values.tolist()

    @property
    def values(self) -> xr.DataArray:
        """Track values, dims (position, <column_dim>)."""
        return self.data["values"]

    @property
    def coords(self):
        """Coordinates of the track data (contigs, offsets, columns)"""
        return self.data.coords

    @property
    def attrs(self) -> dict:
        """Attributes of the track group."""
        return self.data.attrs

    def column(self, label: str) -> xr.DataArray:
        """Values of one column, dims (position,)."""
        return self.values.sel({self.column_dim: label})

    def _require_sum(self) -> Track:
        if self.summaries.sum is None:
            raise MissingSummaryError(
                f"track {self.name!r} has no sum summary; run `pbview preprocess`"
            )
        return self.summaries.sum

    def require_missingness(self, cutoff: int) -> Track:
        """Missingness summary for `cutoff`, or a `MissingSummaryError`."""
        summary = self.summaries.missingness.get(cutoff)
        if summary is None:
            available = sorted(self.summaries.missingness)
            raise MissingSummaryError(
                f"track {self.name!r} has no missingness summary for cutoff "
                f"{cutoff} (available: {available}); run `pbview preprocess "
                f"missingness --missing-data-threshold {cutoff}`"
            )
        return summary

    def sum_hist(
        self, sample_set: str, n_samples: int, max_positions: int = 0
    ) -> np.ndarray:
        """`sum_hists` for one sample set."""
        return self.sum_hists((sample_set,), (n_samples,), max_positions)[sample_set]

    @lru_cache(maxsize=None)
    def sum_hists(
        self,
        sample_sets: tuple[str, ...],
        n_samples: tuple[int, ...],
        max_positions: int = 0,
    ) -> dict[str, np.ndarray]:
        """Histograms over the genome of the summed coverage of sample sets.

        `n_samples[i]` is the number of samples in `sample_sets[i]`. All
        histograms are computed in one pass over the data: the store is
        read once, not once per sample set (about 1.3 times faster for
        four sets on a 660 Mbp store; the chunks of the sum track hold all
        sets).

        Sums above the cap (`n_samples * PBVIEW_MAX_COVERAGE_PER_SAMPLE`,
        at most `PBVIEW_MAX_COVERAGE_BIN`) are clipped into the last bin,
        which is then the cap itself. Trailing empty bins are dropped, so
        the length is the largest observed (clipped) sum plus one.

        With `max_positions` (> 0) and more positions than that, every
        k-th position is counted, k = ceil(positions / max_positions),
        so the counts are those of a sample and sum to the number of
        positions counted, not to the genome size. Every chunk is still
        read; the saving is in the counting. Coverage has no periodicity
        at this scale, so the sample is representative (on a 660 Mbp
        store with 4 sample sets, counting every 66th position moved
        lower/upper/max_bin by less than 0.3%). Reading all sets once and
        sampling together made the thresholds 3.6 times faster there.

        Returns:
            dict mapping each sample set to its histogram
        """
        if len(sample_sets) != len(n_samples):
            raise ValueError("sample_sets and n_samples must have the same length")
        summary = self._require_sum()
        caps = [
            min(n * config.MAX_COVERAGE_PER_SAMPLE, config.MAX_COVERAGE_BIN)
            for n in n_samples
        ]
        columns = [summary.column(s).data for s in sample_sets]
        n = columns[0].shape[0] if columns else 0
        stride = -(-n // max_positions) if 0 < max_positions < n else 1
        if stride > 1:
            logger.info(
                "Estimating the sample set coverage histograms from every %d-th "
                "position (%d of %d)",
                stride,
                -(-n // stride),
                n,
            )
            columns = [c[::stride] for c in columns]
        graphs = [
            da.bincount(da.minimum(c, cap), minlength=cap + 1)
            for c, cap in zip(columns, caps)
        ]
        hists = dask.compute(*graphs)
        result = {}
        for s, hist, cap in zip(sample_sets, hists, caps):
            if len(hist) > cap and hist[cap] > 0:
                logger.warning(
                    "%s: %d counted positions have summed coverage >= the "
                    "configured maximum (%d); they are counted in the last bin.",
                    s,
                    int(hist[cap]),
                    cap,
                )
            nonzero = np.flatnonzero(hist)
            result[s] = hist[: nonzero[-1] + 1] if len(nonzero) else hist[:1]
        return result

    @lru_cache(maxsize=None)
    def contig_mean_coverage(self, sample_set: str) -> np.ndarray:
        """Mean summed coverage per contig for a sample set."""
        # FIXME: make progress optional
        with ProgressBar():
            return self._compute_contig_mean(sample_set)

    def _compute_contig_mean(self, sample_set: str) -> np.ndarray:
        logger.info("Calculating mean contig coverages")
        summary = self._require_sum()
        sums_da = summary.column(sample_set).data
        offsets = summary.coords["offsets"].values
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

    def select(self, resolved: ResolvedSelection) -> xr.Dataset:
        """Select the resolved columns and positions (original chunking)."""
        if resolved.column_dim != self.column_dim:
            raise ValueError(
                f"selection resolved for column dim {resolved.column_dim!r}, "
                f"track {self.name!r} has {self.column_dim!r}"
            )
        return select_data(self.data, resolved)

    def _optimal_chunks(self, reduce_along: str) -> dict:
        """Return optimal chunk sizes for a dask array based on target bytes."""
        logger.debug(
            "Calculating optimal chunks for target bytes: %s", self.TARGET_BYTES
        )
        return optimal_chunks_dask(self.data["values"], reduce_along, self.TARGET_BYTES)

    @property
    def _hist_view(self):
        """Rechunk data for histogram calculations.
        Ensure chunks are across all samples."""
        logger.debug("Rechunking data for histogram calculations")
        return self.data.chunk(self._optimal_chunks(self.column_dim))

    @lru_cache(maxsize=64)
    def _coverage_hist_cached(
        self, bins: npt.NDArray[int], resolved: ResolvedSelection
    ):
        logger.info("Calculating histogram of coverage for '%s' track", self.name)
        logger.debug("Current selection: %s", resolved)
        bins = np.asarray(bins)
        name = self._matching_group_of(resolved.labels, "sample_set")
        summary = self.summaries.sum
        if name is not None and summary is not None:
            sums = _isel_intervals(summary.column(name), resolved.intervals)
        else:
            # Raw path
            arr = select_data(self._hist_view, resolved)["values"]
            sums = arr.sum(self.column_dim).astype(np.int32)
        max_bin = int(bins[-1])
        sums_clipped = da.minimum(sums.data, max_bin)
        hist = da.bincount(sums_clipped, minlength=int(bins[-1]) + 1)
        with ProgressBar():
            return hist.compute(), bins

    def coverage_hist(
        self,
        bins: npt.NDArray[int],
        resolved: ResolvedSelection,
    ):
        """Calculate coverage histogram for a given number of bins and selection.

        The histogram values are calculated as the sum of values across all samples.

        Args:
            bins: number of bins for histogram
            resolved: selection resolved for this track
                (`Coordinates.resolve_for`)
        Returns:
            Histogram, bins as a numpy arrays
        """
        return self._coverage_hist_cached(tuple(bins.tolist()), resolved)

    @lru_cache(maxsize=64)
    def _missingness_hist_cached(
        self, bins: npt.NDArray, resolved: ResolvedSelection, missing_cutoff: int
    ):
        logger.info("Calculating missingness histogram for '%s' track", self.name)
        logger.debug("Current selection: %s", resolved)
        bins = np.asarray(bins)
        name = self._matching_group_of(resolved.labels, "sample_set")

        summary = self.summaries.missingness.get(missing_cutoff)
        if name is not None and summary is not None:
            counts = _isel_intervals(summary.column(name), resolved.intervals).data
        else:
            # Raw path
            arr = select_data(self._hist_view, resolved)["values"]
            counts = (arr <= missing_cutoff).astype(np.int32).sum(self.column_dim).data
        max_bin = int(bins[-1])
        counts_clipped = da.minimum(counts, max_bin)
        with ProgressBar():
            hist = da.bincount(counts_clipped, minlength=max_bin).compute()
        return hist, bins

    def missingness_hist(
        self,
        resolved: ResolvedSelection,
        missing_cutoff: int = 0,
    ):
        """Calculate missingness histogram for a given number of bins
        and selection.

        The histogram values are calculated as the number of values across all
        samples greater than missing_cutoff.

        Args:
            resolved: selection resolved for this track
                (`Coordinates.resolve_for`)
            missing_cutoff: cutoff value to treat values as missing (default: 0)
        Returns:
            Histogram, bins as a numpy arrays
        """
        bins = np.arange(0, len(resolved.labels) + 1)
        return self._missingness_hist_cached(
            tuple(bins.tolist()), resolved, missing_cutoff
        )

    # FIXME: would it be possible to cheaply calculate the *combined*
    # effect of coverage and missing data filters? This would require
    # applying selected thresholds to *all* positions jointly and
    # returning the .dims["position"] attribute. Modifying the
    # thresholds for the individual histograms is cheap; the joint
    # operation is not. Would need reactive "UPDATE" button.
    def size(self):
        pass


def find_summaries(tracks: Iterable[Track]) -> dict[str, Summaries]:
    """Group summary tracks by their source track.

    Several summaries of one kind for the same source are ambiguous:
    the first by track name is kept and the others are logged.
    """
    sums: dict[str, Track] = {}
    miss: dict[str, dict[int, Track]] = {}

    def keep(found: dict, key, track: Track):
        if key in found:
            logger.warning(
                "Ignoring summary %r: %r already summarizes the same source",
                track.name,
                found[key].name,
            )
        else:
            found[key] = track

    for t in sorted(tracks, key=lambda t: t.name):
        source = t.attrs.get("perbase:source_track")
        kind = t.attrs.get("perbase:provenance")
        if source is None:
            continue
        if kind == config.SUMMARY_SUM:
            keep(sums, source, t)
        elif kind == config.SUMMARY_MISSINGNESS and "perbase:missing_cutoff" in t.attrs:
            keep(miss.setdefault(source, {}), int(t.attrs["perbase:missing_cutoff"]), t)
    return {s: Summaries(sums.get(s), miss.get(s, {})) for s in {*sums, *miss}}
