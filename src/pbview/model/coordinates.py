"""
Coordinates class and helper functions.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from .datastore import DataStore

from pbview._compat import _warn_shim

xr.set_options(display_expand_attrs=False)


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
    selected: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.selected is not None and len(self.selected) == 0:
            object.__setattr__(self, "selected", None)

    def apply(self, data: xr.Dataset):
        """Apply the selection to the given dataset."""
        if self.selected is None:
            return data
        return data.sel({self.column_dim: list(self.selected)})

    def project_to(
        self, target_dim: str, grouping: xr.DataArray | None
    ) -> "ColumnSelection":
        """Project a selection to a target dimension using a grouping.

        The grouping maps values from the current column dimension to
        the target dimension.
        """
        if self.selected is None:
            return ColumnSelection(column_dim=target_dim, selected=None)

        if target_dim == self.column_dim:
            return self

        if grouping is None:
            return ColumnSelection(column_dim=target_dim, selected=None)

        if self.column_dim in grouping.dims and target_dim in grouping.dims:
            # Map selected values to target dimension using grouping
            selected_indices = grouping.sel({self.column_dim: list(self.selected)}).any(
                dim=self.column_dim
            )
            projected_values = grouping.coords[target_dim].values[selected_indices]
            return ColumnSelection(
                column_dim=target_dim, selected=tuple(projected_values)
            )

        return ColumnSelection(column_dim=target_dim, selected=None)


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
class Coordinates:
    """Immutable class representation of active sample/contig
    coordinates.

    Representation of samples, contigs and contig lengths. Provides
    convenience functions to subset coordinates and to keep track of
    active samples and contigs. Every filter operation returns a *new*
    `Coordinates` object. The derived quantities are cached per
    instance.

    The full coordinate set is saved in private variables, meaning a
    `Coordinates` object always has access to the full data view.

    Variable and function names with suffix `_all` refer to the full
    coordinate set, while those without suffix refer to the active
    subset.

    """

    # FIXME: new conversation emphasizes that Coordinates should *not*
    # hold a referenc to the datastore; this breaks meaningful == and
    # hashing, and every with_* carries I/O state,
    datastore: DataStore = field(repr=False, compare=False)  # SHIM: remove
    spatial: SpatialSelection = SpatialSelection()
    column: dict[str, ColumnSelection] = field(default_factory=dict)
    _sample_selected: npt.NDArray[np.bool_] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        sample = self._sample_selected
        object.__setattr__(
            self,
            "_sample_selected",
            np.ones(self.n_samples_all, dtype=bool)
            if sample is None
            else np.asarray(sample, dtype=bool),
        )

    def _key(self) -> tuple:
        return (
            self.datastore.id,  # SHIM: remove with datastore
            self.spatial,  # final
            self._sample_selected.tobytes(),  # SHIM: remove with samples
        )

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other):
        if not isinstance(other, Coordinates):
            return NotImplemented
        return self._key() == other._key()

    def _replace(
        self,
        *,
        sample_selected: npt.NDArray[np.bool_] | None = None,
    ) -> "Coordinates":
        """Return a new instance sharing raw data, with new selection arrays."""
        changes = {}
        if sample_selected is not None:
            changes["_sample_selected"] = np.asarray(sample_selected, dtype=bool)
        new = replace(self, **changes)
        return new

    def matching_sample_set(self) -> str | None:
        """Return the name of the sample set that matches the current sample
        selection mask."""
        for name in self.sample_set_names:
            if name == "ALL":
                members = self.samples_all
            else:
                members = self.samples_all[self.sample_set_membership_all == name]
            if np.array_equal(np.sort(members), np.sort(self.samples)):
                return name
        return None

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

    def with_samples(
        self,
        samples: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed samples. `None` resets the sample selected mask."""
        if samples is None:
            return self._replace(sample_selected=np.ones_like(self._sample_selected))
        keep = (
            np.isin(self.samples_all, np.asarray(list(samples))) & self._sample_selected
        )
        return self._replace(sample_selected=keep)

    def with_sample_sets(self, sample_sets: list[str]):
        """Keep only listed sample sets. `None` resets the sample selected mask."""
        if self.default_sample_set_name in sample_sets:
            return self
        if sample_sets is None:
            return self._replace(sample_selected=np.ones_like(self._sample_selected))
        keep = (
            np.isin(self.sample_set_membership_all, sample_sets) & self._sample_selected
        )
        return self._replace(sample_selected=keep)

    def reset(self) -> "Coordinates":
        """Return Coordinates with no selections (all contigs, all columns)."""
        return type(self)(
            datastore=self.datastore
        )  # SHIM: becomes Coordinates() in (f)

    @property
    def samples_all(self) -> npt.NDArray:
        """All samples in the dataset"""
        return self.datastore.samples

    @property
    def n_samples_all(self) -> int:
        return self.datastore.n_samples

    @property
    def offsets(self) -> npt.NDArray:
        """Contig offsets"""
        return self.datastore.offsets

    @property
    def sample_selected(self):  # SHIM
        _warn_shim("Coordinates.sample_selected", "resolve(track)")
        return self._sample_selected

    @property
    def sample_excluded(self):
        return ~self._sample_selected

    def selected_contigs(self, genome: Genome) -> tuple[str, ...]:
        """Currently selected contigs"""
        return genome.resolve_contigs(self.spatial)

    @property
    def samples(self) -> Any:
        """Return selected samples"""
        return self.samples_all[self._sample_selected]

    @property
    def n_samples(self):
        return int(self.samples.size)

    @property
    def sample_set_membership(self):
        """Return sample sets per sample"""
        return self.datastore.sample_set_membership[self._sample_selected]

    @property
    def sample_set_membership_all(self):
        """Return sample sets per sample"""
        return self.datastore.sample_set_membership

    @property
    def default_sample_set_membership(self):
        return self.datastore.default_sample_set_membership[self._sample_selected]

    @property
    def default_sample_set_membership_all(self):
        return self.datastore.default_sample_set_membership

    @property
    def n_sample_sets(self):
        """Return sample sets size, including the default set."""
        return len(self.sample_set_names)

    @property
    def n_user_sample_sets(self):
        """Return user sample sets size for the current selection"""
        return len(self.user_sample_set_names)

    @property
    def n_user_sample_sets_all(self):
        """Return user sample sets size for entire dataset"""
        return len(self.user_sample_set_names_all)

    @property
    def user_sample_set_names(self) -> list[str]:
        """Return user sample set names for the current selection"""
        if not self.has_sample_sets:
            return []
        selected_members = self.sample_set_membership_all[self._sample_selected]
        present = np.unique(selected_members).tolist()
        return [s for s in present if s != self.default_sample_set_name]

    @property
    def user_sample_set_names_all(self) -> list[str]:
        return self.datastore.user_sample_set_names

    @property
    def sample_set_names(self) -> list[str]:
        return [self.default_sample_set_name, *self.user_sample_set_names]

    @property
    def sample_set_names_all(self) -> list[str]:
        return self.datastore.sample_set_names

    @property
    def default_sample_set_name(self) -> str:
        return self.datastore.default_sample_set_name

    @property
    def has_sample_sets(self):
        """Return True if there are any sample sets other than the default set."""
        return self.datastore.has_sample_sets

    def sample_set_membership_dataframe(self, *, active_only=False) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "sample_set": self.default_sample_set_membership_all,
                "active": self._sample_selected,
                "sample": self.samples_all,
            }
        )
        if self.has_sample_sets:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "sample_set": self.sample_set_membership_all,
                            "active": self._sample_selected,
                            "sample": self.samples_all,
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
