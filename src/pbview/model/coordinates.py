"""
Selection values and the universes they are resolved against.

`SpatialSelection` and `ColumnSelection` are pure, hashable values: `None`
means "no filter" (the whole universe) and an empty tuple means "nothing".
A `Coordinates` combines one spatial selection with at most one column
selection per column dimension. `Coordinates.resolve_for(track, genome)`
resolves it against a `Genome` and a `Track` into a `ResolvedSelection` with
explicit intervals and labels, which `Track.select` applies to the data.
Validation (unknown contigs or labels) happens when a selection is
resolved, never at construction.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .track import Track

from pbview import config

xr.set_options(display_expand_attrs=False)


# Projection policy: (selection dim, track dim) pairs that project through a grouping.
_PROJECTIONS = frozenset({("sample_set", "sample")})


def resolve_columns(
    selections: Iterable[ColumnSelection],
    column_dim: str,
    universe: Sequence[str],
    groupings: Mapping[str, xr.DataArray] = MappingProxyType({}),
) -> tuple[str, ...]:
    """Labels of `column_dim` selected by `selections`, in universe order.

    Selections on different dimensions intersect. A selection on the
    track's own column dimension picks labels directly; one on a dimension
    with a grouping (for example `sample_set` on a `sample` track) picks
    every column that belongs to any selected group. A selection that does
    not apply to the track is skipped, except for the projection pairs in
    `_PROJECTIONS`, where a missing grouping is an error. Unknown labels
    raise a `ValueError`.
    """
    constraints: list[set[str]] = []
    for sel in selections:
        if sel.labels is None:
            continue
        if sel.column_dim == column_dim:
            _check_known(sel.labels, universe, column_dim)
            constraints.append(set(sel.labels))
            continue
        grouping = groupings.get(sel.column_dim)
        if grouping is None:
            if (sel.column_dim, column_dim) in _PROJECTIONS:
                raise ValueError(
                    f"no {sel.column_dim!r} grouping for the {column_dim!r} track"
                )
            continue  # the selection does not apply to this track
        groups = grouping[sel.column_dim].values
        _check_known(sel.labels, groups, sel.column_dim)
        cols = np.isin(groups, sel.labels)
        rows = grouping.values[:, cols].any(axis=1)
        constraints.append(set(grouping[column_dim].values[rows]))
    if not constraints:
        return tuple(universe)
    keep = set.intersection(*constraints)
    return tuple(c for c in universe if c in keep)


def _check_known(labels, known, dim):
    unknown = set(labels) - set(known)
    if unknown:
        raise ValueError(f"unknown {dim} labels: {sorted(unknown)}")


@dataclass(frozen=True)
class SpatialSelection:
    """A selection over the spatial axis of a track.

    The spatial axis (contig, position) is a global coordinate system,
    meaning that the same selection can be applied to any track.

    Only the contig range can be directly used for selection and will
    be resolved for in the Genome class.

    Future versions may support region-like selection of contig,
    start, end tuples; resolution will still be handled by the Genome
    class.

    """

    contigs: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.contigs is None:
            return None
        if isinstance(self.contigs, str):
            raise TypeError("contigs must be a sequence of names, not a str")
        object.__setattr__(self, "contigs", tuple(self.contigs))


@dataclass(frozen=True)
class ColumnSelection:
    """A selection over the column axis of a track."""

    column_dim: str
    labels: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.labels is None:
            return
        if isinstance(self.labels, str):
            raise TypeError("selected labels must be a sequence, not a str")
        object.__setattr__(self, "labels", tuple(self.labels))

    def apply(self, data: xr.Dataset):
        """Apply the selection to the given dataset."""
        if self.labels is None:
            return data
        return data.sel({self.column_dim: list(self.labels)})

    def project_to(
        self, target_dim: str, grouping: xr.DataArray | None
    ) -> "ColumnSelection":
        """Project a selection to a target dimension using a grouping.

        The grouping maps values from the current column dimension to
        the target dimension.
        """
        if self.labels is None:
            return ColumnSelection(column_dim=target_dim, labels=None)

        if target_dim == self.column_dim:
            return self

        if grouping is None:
            return ColumnSelection(column_dim=target_dim, labels=None)

        if self.column_dim in grouping.dims and target_dim in grouping.dims:
            # Map selected values to target dimension using grouping
            selected_indices = grouping.sel({self.column_dim: list(self.labels)}).any(
                dim=self.column_dim
            )
            projected_values = grouping.coords[target_dim].values[selected_indices]
            return ColumnSelection(
                column_dim=target_dim, labels=tuple(projected_values)
            )

        return ColumnSelection(column_dim=target_dim, labels=None)


def _merge_adjacent(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[tuple[int, int], ...]:
    """Merge sorted, disjoint intervals"""
    if starts.size == 0:
        return ()
    breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
    run_starts = starts[np.r_[0, breaks]]
    run_ends = ends[np.r_[breaks - 1, starts.size - 1]]
    return tuple(zip(run_starts.tolist(), run_ends.tolist()))


@dataclass(frozen=True)
class Genome:
    """Contigs and their offsets (CSR style, `len(contigs) + 1` entries).

    Maps contig selections to global position intervals. It holds tuples
    only, so it is hashable and can key `lru_cache`.
    """

    contigs: tuple[str, ...]
    offsets: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "contigs", tuple(self.contigs))
        object.__setattr__(self, "offsets", tuple(int(o) for o in self.offsets))
        if len(self.contigs) + 1 != len(self.offsets):
            raise ValueError(
                "Offsets must be one longer than contigs, got "
                f"{len(self.contigs)} contigs and {len(self.offsets)} offsets"
            )

    @cached_property
    def _offsets_arr(self) -> np.ndarray:
        return np.asarray(self.offsets, dtype=np.int64)

    @property
    def size(self):
        """Size of genome in bp"""
        return self.offsets[-1] - self.offsets[0]

    @cached_property
    def _contig_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.contigs)}

    # FIXME: this function is obsolete (replaced by _contig_index)
    def _index(self, contig: str) -> int:
        try:
            return self.contigs.index(contig)
        except ValueError:
            raise KeyError(f"unknown contig {contig!r}")

    def contig_indices(self, contigs: Iterable[str] | None = None) -> np.ndarray:
        if contigs is None:
            return np.arange(len(self.contigs), dtype=np.intp)
        contigs = tuple(contigs)
        unknown = [c for c in contigs if c not in self._contig_index]
        if unknown:
            raise KeyError(f"unknown contigs {unknown}")
        return np.unique(
            np.fromiter(
                (self._contig_index[c] for c in contigs),
                dtype=np.intp,
                count=len(contigs),
            )
        )

    @property
    def lengths(self) -> tuple[int, ...]:
        return np.asarray(self.offsets[1:]) - np.asarray(self.offsets[:-1])

    def contigs_in_length_range(
        self,
        lower: int = 0,
        upper: float = np.inf,
        contigs: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        idx = self.contig_indices(contigs)
        lengths = self.lengths[idx]
        keep = (lengths >= lower) & (lengths <= upper)
        return tuple(self.contigs[i] for i in idx[keep])

    def span(self, contig: str) -> tuple[int, int]:
        """Return the start and end positions for a given contig."""
        idx = self.contigs.index(contig)
        return self.offsets[idx], self.offsets[idx + 1]

    def resolve(self, sel: SpatialSelection) -> "PositionIntervals":
        """Resolve selection ranges"""
        idx = np.unique(self.contig_indices(sel.contigs))
        return PositionIntervals(
            _merge_adjacent(self._offsets_arr[idx], self._offsets_arr[idx + 1])
        )

    def resolve_contigs(self, spatial: SpatialSelection) -> tuple[str, ...]:
        """Resolve contig names"""
        if spatial.contigs is None:
            return self.contigs
        return tuple(self.contigs[i] for i in self.contig_indices(spatial.contigs))

    # FIXME: Convert to generator if list is long?
    def contig_regions(self, sel: SpatialSelection) -> tuple[tuple[str, int, int], ...]:
        """Selected regions in contig-local, 0-based half-open coordinates"""
        idx = self.contig_indices(sel.contigs)
        return tuple((self.contigs[i], 0, int(self.lengths[i])) for i in idx)


@dataclass(frozen=True)
class PositionIntervals:
    """A collection of position intervals, represented as a tuple of (start, end) pairs.

    PositionIntervals represent intervals on the position axis of a
    track. They know nothing of contig space, and are simply a
    collection of (start, end) pairs. They can be used to select data
    from a track, and can be converted to a PyRanges object for
    further manipulation.

    """

    intervals: tuple[tuple[int, int], ...]

    @property
    def n_positions(self) -> int:
        """Return the total interval length"""
        if len(self.intervals) == 0:
            return 0
        return sum([e - s for s, e in self.intervals])

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        parts = [data.isel(position=slice(s, e)) for s, e in self.intervals]
        return parts[0] if len(parts) == 1 else xr.concat(parts, dim="position")


@dataclass(frozen=True)
class ResolvedSelection:
    """A selection resolved for one track and genome: explicit and hashable.

    `labels` are the selected columns of the track's column dimension in
    track order. An empty tuple means no columns, never "all".
    """

    intervals: PositionIntervals
    column_dim: str
    labels: tuple[str, ...]  # explicit; empty means empty


@dataclass(frozen=True)
class Coordinates:
    """Immutable class representation of active sample/contig
    coordinates.

    A pure value: a `SpatialSelection` and at most one
    `ColumnSelection` per column dimension. Every `with_*` method
    returns a *new* `Coordinates` object. Selections are resolved
    against their universes, `Genome` and `Track`, never here.
    """

    spatial: SpatialSelection = SpatialSelection()
    column: tuple[ColumnSelection, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "column", tuple(self.column))
        dims = [c.column_dim for c in self.column]
        if len(dims) != len(set(dims)):
            raise ValueError(f"duplicate column dims: {dims}")

    def column_for(self, dim: str) -> ColumnSelection:
        """Return column selection for dim"""
        return next(
            (c for c in self.column if c.column_dim == dim), ColumnSelection(dim)
        )

    def resolve_for(self, track: Track, genome: Genome) -> ResolvedSelection:
        """Resolve coordinates with respect to a given track and genome"""
        return ResolvedSelection(
            genome.resolve(self.spatial),
            track.column_dim,
            track.resolve_columns(self.column),
        )

    # Factory methods
    def with_contigs(
        self,
        contigs: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Select contigs. None or empty means the whole genome."""
        return replace(
            self,
            spatial=SpatialSelection(contigs),
        )

    def with_columns(
        self,
        selection: ColumnSelection,
    ) -> "Coordinates":
        """Replace the selection for selection.column_dim"""
        others = tuple(s for s in self.column if s.column_dim != selection.column_dim)
        keep = others if selection.labels is None else others + (selection,)
        return replace(self, column=keep)

    def with_samples(
        self,
        samples: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only the listed samples, replacing any earlier sample selection.

        `None` resets the `sample` selection. Unknown labels are rejected when
        the selection is resolved.
        """
        labels = None if samples is None else tuple(samples)
        return self.with_columns(ColumnSelection("sample", labels))

    def with_sample_sets(
        self, sample_sets: Iterable[str] | None = None
    ) -> "Coordinates":
        """Keep samples in any of the listed sets, replacing any
        earlier set selection.

        `None`, or a list containing the default set, resets the
        `sample_set` selection. Unknown labels are rejected when the
        selection is resolved.
        """
        labels = None if sample_sets is None else tuple(sample_sets)
        if labels is not None and config.DEFAULT_GROUPING_SET in labels:
            labels = None
        return self.with_columns(ColumnSelection("sample_set", labels))

    def selected_contigs(self, genome: Genome) -> tuple[str, ...]:
        """Currently selected contigs"""
        return genome.resolve_contigs(self.spatial)

    def selected_columns(self, track: Track) -> tuple[str, ...]:
        """Currently selected columns"""
        return track.resolve_columns(self.column)
