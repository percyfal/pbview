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
from .histogram import fold_hist
from .pca import PcaResult, pca_from_moments, window_moments, window_starts

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


def sample_stride(n_positions: int, max_positions: int) -> int:
    """Every k-th position is counted when `max_positions` is exceeded.

    k = ceil(n_positions / max_positions); 1 means all positions
    (`max_positions` <= 0, or no more positions than that).
    """
    if 0 < max_positions < n_positions:
        return -(-n_positions // max_positions)
    return 1


def _scaled(counts: np.ndarray, n_total: int, n_counted: int) -> np.ndarray:
    """Scale counts of a sample of positions up to `n_total` positions."""
    if n_counted == 0 or n_counted == n_total:
        return counts
    return np.rint(counts * (n_total / n_counted)).astype(counts.dtype)


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

    def _summary_sets(self, summary: Track) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Sample sets of a summary track and their number of samples."""
        totals = self.group_counts("sample_set", ())
        sets = tuple(summary.columns)
        return sets, tuple(totals[s][1] for s in sets)

    @staticmethod
    def _count_hists(
        summary: Track,
        sample_sets: tuple[str, ...],
        caps: list[int],
        max_positions: int,
        intervals: PositionIntervals | None,
    ) -> tuple[list[np.ndarray], int]:
        """Histograms of summary columns in one pass: (histograms, positions counted).

        Values are clipped at `caps[i]`, so histogram i has at least
        `caps[i] + 1` bins. Every k-th position is counted, see
        `sample_stride`.
        """
        columns = []
        for s in sample_sets:
            column = summary.column(s)
            if intervals is not None:
                column = _isel_intervals(column, intervals)
            columns.append(column.data)
        n = columns[0].shape[0] if columns else 0
        stride = sample_stride(n, max_positions)
        if stride > 1:
            logger.info(
                "Estimating histograms from every %d-th position (%d of %d)",
                stride,
                -(-n // stride),
                n,
            )
            columns = [c[::stride] for c in columns]
        graphs = [
            da.bincount(da.minimum(c, cap), minlength=cap + 1)
            for c, cap in zip(columns, caps)
        ]
        return list(dask.compute(*graphs)), -(-n // stride)

    @lru_cache(maxsize=64)
    def sum_hists(
        self,
        sample_sets: tuple[str, ...],
        n_samples: tuple[int, ...],
        max_positions: int = 0,
        intervals: PositionIntervals | None = None,
    ) -> dict[str, np.ndarray]:
        """Histograms of the summed coverage of sample sets.

        `n_samples[i]` is the number of samples in `sample_sets[i]`. The
        positions are those of `intervals`, default the whole genome. All
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
        positions counted, not to the number of positions. Every chunk is
        still read; the saving is in the counting. Coverage has no
        periodicity at this scale, so the sample is representative (on a
        660 Mbp store with 4 sample sets, counting every 66th position
        moved lower/upper/max_bin by less than 0.3%). Reading all sets
        once and sampling together made the thresholds 3.6 times faster
        there.

        Returns:
            dict mapping each sample set to its histogram
        """
        if len(sample_sets) != len(n_samples):
            raise ValueError("sample_sets and n_samples must have the same length")
        caps = [
            min(n * config.MAX_COVERAGE_PER_SAMPLE, config.MAX_COVERAGE_BIN)
            for n in n_samples
        ]
        hists, _ = self._count_hists(
            self._require_sum(), sample_sets, caps, max_positions, intervals
        )
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

    @lru_cache(maxsize=64)
    def missingness_hists(
        self,
        cutoff: int,
        sample_sets: tuple[str, ...],
        n_samples: tuple[int, ...],
        max_positions: int = 0,
        intervals: PositionIntervals | None = None,
    ) -> dict[str, np.ndarray]:
        """Histograms of the number of missing samples of sample sets.

        As `sum_hists`, for the missingness summary of `cutoff`. Histogram
        i has exactly `n_samples[i] + 1` bins: 0 .. all samples missing.
        """
        if len(sample_sets) != len(n_samples):
            raise ValueError("sample_sets and n_samples must have the same length")
        hists, _ = self._count_hists(
            self.require_missingness(cutoff),
            sample_sets,
            list(n_samples),
            max_positions,
            intervals,
        )
        return dict(zip(sample_sets, hists))

    def contig_mean_coverage(self, sample_set: str) -> np.ndarray:
        """Mean summed coverage per contig for a sample set."""
        return self._contig_means()[sample_set]

    @lru_cache(maxsize=None)
    def _contig_means(self) -> dict[str, np.ndarray]:
        """Mean summed coverage per contig of every sample set, in one pass."""
        # FIXME: make progress optional
        with ProgressBar():
            return self._compute_contig_means()

    def _compute_contig_means(self) -> dict[str, np.ndarray]:
        logger.info("Calculating mean contig coverages")
        summary = self._require_sum()
        sums = summary.values.transpose("position", summary.column_dim)
        sums_da = sums.data.rechunk({1: -1})
        offsets = summary.coords["offsets"].values
        lengths = np.diff(offsets)
        n_contigs = len(lengths)
        n_sets = sums_da.shape[1]

        def _partial_reduceat(chunk, block_info=None):
            start = block_info[0]["array-location"][0][0]
            end = block_info[0]["array-location"][0][1]
            partial = np.zeros((1, n_contigs, n_sets), dtype=np.int64)
            first = np.searchsorted(offsets, start, side="right") - 1
            last = np.searchsorted(offsets, end, side="left")
            local_offsets = np.clip(offsets[first : last + 1] - start, 0, len(chunk))
            if len(local_offsets) < 2:
                return partial
            reduced = np.add.reduceat(chunk, local_offsets[:-1], axis=0, dtype=np.int64)
            partial[0, first : first + len(reduced)] = reduced
            return partial

        per_chunk = sums_da.map_blocks(
            _partial_reduceat,
            dtype=np.int64,
            chunks=((1,) * sums_da.numblocks[0], (n_contigs,), (n_sets,)),
            new_axis=1,
        )
        per_contig_sum = per_chunk.sum(axis=0).compute()
        means = per_contig_sum / lengths[:, None]
        return {s: means[:, i] for i, s in enumerate(summary.columns)}

    def group_membership(self, dim: str = "sample_set") -> dict[str, tuple[str, ...]]:
        """The groups of `dim` that each column belongs to.

        `ALL` is left out for a column that is in another group; a column in
        no group of the track's grouping gets `ALL`.
        """
        grouping = self._grouping_for(dim)
        names = grouping[grouping.dims[1]].values.tolist()
        member = grouping.values
        result = {}
        for j, column in enumerate(self.columns):
            groups = [n for n, m in zip(names, member[j]) if m]
            specific = [n for n in groups if n != config.DEFAULT_GROUPING_SET]
            result[column] = tuple(specific or groups or [config.DEFAULT_GROUPING_SET])
        return result

    @lru_cache(maxsize=16)
    def pca(self, resolved: ResolvedSelection, window_size: int) -> PcaResult:
        """PCA of the selected samples, with the samples as the features.

        The data are the mean coverage of the selected samples in windows of
        `window_size` positions inside the selected contigs, scaled by the
        mean coverage of each sample (see `pbview.model.pca`). It makes one
        pass over the raw values, reading only the chunks that hold windows,
        and keeps only n_samples x n_samples numbers in memory.

        Args:
            resolved: selection resolved for this track
                (`Coordinates.resolve_for`)
            window_size: window length in positions

        Raises:
            ValueError: if there are fewer than two samples or windows
        """
        starts = window_starts(
            self.coords["offsets"].values, resolved.intervals, window_size
        )
        values = self.values.sel({self.column_dim: list(resolved.labels)})
        logger.info(
            "PCA of %d samples in %d windows of %d positions",
            len(resolved.labels),
            len(starts),
            window_size,
        )
        with ProgressBar():
            moments = window_moments(values.data, starts, window_size)
        return pca_from_moments(moments, resolved.labels, window_size)

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
        self,
        bins: npt.NDArray[int],
        resolved: ResolvedSelection,
        max_positions: int = 0,
    ):
        logger.info("Calculating histogram of coverage for '%s' track", self.name)
        logger.debug("Current selection: %s", resolved)
        bins = np.asarray(bins)
        name = self._matching_group_of(resolved.labels, "sample_set")
        summary = self.summaries.sum
        max_bin = int(bins[-1])
        n_total = resolved.intervals.n_positions
        if name is not None and summary is not None:
            sets, sizes = self._summary_sets(summary)
            cap = min(
                sizes[sets.index(name)] * config.MAX_COVERAGE_PER_SAMPLE,
                config.MAX_COVERAGE_BIN,
            )
            if max_bin <= cap:
                # All sample sets in one pass; bins are folded from the
                # histogram, so changing them needs no new pass
                h = self.sum_hists(sets, sizes, max_positions, resolved.intervals)[name]
                counts = fold_hist(h, max_bin + 1)
                return _scaled(counts, n_total, int(h.sum())), bins
            data = _isel_intervals(summary.column(name), resolved.intervals).data
            stride = sample_stride(n_total, max_positions)
            if stride > 1:
                data = data[::stride]
            graph = da.bincount(da.minimum(data, max_bin), minlength=max_bin + 1)
            n_counted = data.shape[0]
        else:
            graph, n_counted = self._raw_graph(
                self._hist_view, resolved, max_bin, max_positions
            )
        with ProgressBar():
            counts = graph.compute()
        return _scaled(counts, n_total, n_counted), bins

    def _raw_array(
        self, view: xr.Dataset, resolved: ResolvedSelection, max_positions: int
    ) -> tuple[xr.DataArray, int]:
        """Raw values of a selection and the number of positions in them.

        Every k-th position is kept (see `sample_stride`), before any
        reduction over the columns, so that the skipped positions are
        never summed.
        """
        arr = select_data(view, resolved)["values"]
        stride = sample_stride(resolved.intervals.n_positions, max_positions)
        if stride > 1:
            arr = arr.isel(position=slice(None, None, stride))
        return arr, arr.sizes["position"]

    def _raw_graph(
        self,
        view: xr.Dataset,
        resolved: ResolvedSelection,
        max_bin: int,
        max_positions: int,
        missing_cutoff: int | None = None,
    ) -> tuple[da.Array, int]:
        """Histogram graph over the raw values of a selection, and positions counted.

        The sum of the selected columns, or with `missing_cutoff` the number
        of them at or below it, clipped to `max_bin`.
        """
        arr, n_counted = self._raw_array(view, resolved, max_positions)
        if missing_cutoff is None:
            per_position = arr.sum(self.column_dim).astype(np.int32)
        else:
            per_position = (arr <= missing_cutoff).astype(np.int32).sum(self.column_dim)
        graph = da.bincount(
            da.minimum(per_position.data, max_bin), minlength=max_bin + 1
        )
        return graph, n_counted

    @lru_cache(maxsize=16)
    def _raw_bundle(
        self,
        selections: tuple[tuple[str, ResolvedSelection], ...],
        cutoffs: tuple[int, ...],
        max_positions: int,
    ) -> dict[str, tuple[np.ndarray, dict[int, np.ndarray], int]]:
        """Coverage and missingness histograms of raw selections in one pass.

        For each named selection: the histogram of the summed values, the
        histogram of the number of missing columns for each of `cutoffs`,
        and the number of positions counted. All are computed from the same
        chunk reads, which are what costs: the coverage and the missingness
        pages ask for the same selections, so the second asks for nothing
        new.

        The sums are clipped at the cap of the whole track (all its columns
        times `PBVIEW_MAX_COVERAGE_PER_SAMPLE`, at most
        `PBVIEW_MAX_COVERAGE_BIN`), not at the cap of the selection: the
        page bins come from whole sample sets, and a smaller selection may
        be asked for bins beyond its own cap, which would need another pass.
        """
        view = self._hist_view  # one view, so the graphs share its chunk reads
        cap = min(
            len(self.columns) * config.MAX_COVERAGE_PER_SAMPLE, config.MAX_COVERAGE_BIN
        )
        graphs, layout = [], []
        for name, resolved in selections:
            arr, n_counted = self._raw_array(view, resolved, max_positions)
            n_labels = len(resolved.labels)
            sums = arr.sum(self.column_dim).astype(np.int32).data
            graphs.append(da.bincount(da.minimum(sums, cap), minlength=cap + 1))
            for cutoff in cutoffs:
                missing = (arr <= cutoff).astype(np.int32).sum(self.column_dim).data
                graphs.append(da.bincount(missing, minlength=n_labels + 1))
            layout.append((name, n_counted))
        with ProgressBar():
            hists = list(dask.compute(*graphs))
        result = {}
        for name, n_counted in layout:
            coverage = hists.pop(0)
            missing = {cutoff: hists.pop(0) for cutoff in cutoffs}
            result[name] = (coverage, missing, n_counted)
        return result

    def _raw_many(
        self,
        raw: list[tuple[str, ResolvedSelection, npt.NDArray[int]]],
        missing_cutoff: int | None,
        max_positions: int,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """`coverage_hist` (or `missingness_hist`) of raw selections, via the bundle."""
        cutoffs = set(self.summaries.missingness)
        if missing_cutoff is not None:
            cutoffs.add(missing_cutoff)
        bundle = self._raw_bundle(
            tuple((name, resolved) for name, resolved, _ in raw),
            tuple(sorted(cutoffs)),
            max_positions,
        )
        result = {}
        for name, resolved, bins in raw:
            coverage, missing, n_counted = bundle[name]
            n_total = resolved.intervals.n_positions
            if missing_cutoff is not None:
                counts = fold_hist(missing[missing_cutoff], len(bins))
            elif int(bins[-1]) < len(coverage):  # within the cap
                counts = fold_hist(coverage, int(bins[-1]) + 1)
            else:  # beyond the cap: the bundle is clipped there, count again
                graph, n_counted = self._raw_graph(
                    self._hist_view, resolved, int(bins[-1]), max_positions
                )
                with ProgressBar():
                    counts = graph.compute()
            result[name] = (_scaled(counts, n_total, n_counted), np.asarray(bins))
        return result

    def _hists_of(
        self,
        selections: Mapping[str, ResolvedSelection],
        bins: Mapping[str, npt.NDArray[int]] | None,
        missing_cutoff: int | None,
        max_positions: int,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        precomputed = (
            self.summaries.sum
            if missing_cutoff is None
            else self.summaries.missingness.get(missing_cutoff)
        )
        result, raw = {}, []
        for name, resolved in selections.items():
            sample_bins = (
                bins[name] if bins is not None else np.arange(len(resolved.labels) + 1)
            )
            group = self._matching_group_of(resolved.labels, "sample_set")
            if group is not None and precomputed is not None:
                if missing_cutoff is None:
                    result[name] = self.coverage_hist(
                        sample_bins, resolved, max_positions
                    )
                else:
                    result[name] = self.missingness_hist(
                        resolved, missing_cutoff, max_positions
                    )
            else:
                raw.append((name, resolved, sample_bins))
        if raw:
            result.update(self._raw_many(raw, missing_cutoff, max_positions))
        return {name: result[name] for name in selections}

    def coverage_hist_many(
        self,
        selections: Mapping[str, ResolvedSelection],
        bins: Mapping[str, npt.NDArray[int]],
        max_positions: int = 0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """`coverage_hist` of several named selections.

        Selections that are exactly one sample set use the precomputed
        summary (all sets in one pass, see `sum_hists`). The others sum
        the raw values, and these are computed together in one pass over
        the raw data instead of one per selection. That pass also gives
        their missingness histograms for every cutoff, so a following
        `missingness_hist_many` of the same selections (and the other
        way round) reads nothing.

        Returns:
            {name: (histogram, bins)}
        """
        return self._hists_of(selections, bins, None, max_positions)

    def missingness_hist_many(
        self,
        selections: Mapping[str, ResolvedSelection],
        missing_cutoff: int = 0,
        max_positions: int = 0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """`missingness_hist` of several named selections, as `coverage_hist_many`."""
        return self._hists_of(selections, None, missing_cutoff, max_positions)

    def coverage_hist(
        self,
        bins: npt.NDArray[int],
        resolved: ResolvedSelection,
        max_positions: int = 0,
    ):
        """Calculate coverage histogram for a given number of bins and selection.

        The histogram values are calculated as the sum of values across all samples.
        Coverage above the last bin is counted in it.

        With `max_positions` (> 0) and more selected positions than that,
        every k-th position is counted and the counts are scaled up to
        the number of selected positions, so they are estimates (see
        `sample_stride`, `sum_hists`).

        Args:
            bins: number of bins for histogram
            resolved: selection resolved for this track
                (`Coordinates.resolve_for`)
            max_positions: most positions to count; 0 counts all
        Returns:
            Histogram, bins as a numpy arrays
        """
        return self._coverage_hist_cached(tuple(bins.tolist()), resolved, max_positions)

    @lru_cache(maxsize=64)
    def _missingness_hist_cached(
        self,
        bins: npt.NDArray,
        resolved: ResolvedSelection,
        missing_cutoff: int,
        max_positions: int = 0,
    ):
        logger.info("Calculating missingness histogram for '%s' track", self.name)
        logger.debug("Current selection: %s", resolved)
        bins = np.asarray(bins)
        name = self._matching_group_of(resolved.labels, "sample_set")
        n_total = resolved.intervals.n_positions

        summary = self.summaries.missingness.get(missing_cutoff)
        if name is not None and summary is not None:
            sets, sizes = self._summary_sets(summary)
            h = self.missingness_hists(
                missing_cutoff, sets, sizes, max_positions, resolved.intervals
            )[name]
            return _scaled(fold_hist(h, len(bins)), n_total, int(h.sum())), bins
        graph, n_counted = self._raw_graph(
            self._hist_view, resolved, int(bins[-1]), max_positions, missing_cutoff
        )
        with ProgressBar():
            counts = graph.compute()
        return _scaled(counts, n_total, n_counted), bins

    def missingness_hist(
        self,
        resolved: ResolvedSelection,
        missing_cutoff: int = 0,
        max_positions: int = 0,
    ):
        """Calculate missingness histogram for a given number of bins
        and selection.

        The histogram values are calculated as the number of values across all
        samples greater than missing_cutoff.

        Args:
            resolved: selection resolved for this track
                (`Coordinates.resolve_for`)
            missing_cutoff: cutoff value to treat values as missing (default: 0)
            max_positions: most positions to count, 0 counts all; with
                more selected positions every k-th is counted and the
                counts are scaled up (see `coverage_hist`)
        Returns:
            Histogram, bins as a numpy arrays
        """
        bins = np.arange(0, len(resolved.labels) + 1)
        return self._missingness_hist_cached(
            tuple(bins.tolist()), resolved, missing_cutoff, max_positions
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
