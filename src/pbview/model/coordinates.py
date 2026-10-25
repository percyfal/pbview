"""
Coordinates class and helper functions.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyranges1 as pr
import xarray as xr

if TYPE_CHECKING:
    from .datastore import DataStore


xr.set_options(display_expand_attrs=False)


# FIXME: spatial selections need genome offsets so apply cannot work.
# Therefore, this Protocol is not really useful since only
# ColumnSelection uses apply.
class Selection(Protocol):
    """A protocol for selection over an axis of a track."""

    def apply(self, data: xr.Dataset) -> xr.Dataset: ...


@dataclass(frozen=True)
class SpatialSelection:
    """A selection over the spatial axis of a track.

    The spatial axis (contig, position) is a global coordinate system,
    meaning that the same selection can be applied to any track.

    Only the position range can be directly used for selection.

    """

    contigs: list[str] | None = None
    position_intervals: list[tuple[int, int]] | None = None

    def __post_init__(self):
        if self.position_intervals is not None and len(self.position_intervals) == 0:
            object.__setattr__(self, "position_intervals", None)
        if self.contigs is not None and len(self.contigs) == 0:
            object.__setattr__(self, "contigs", None)

    # FIXME: Remove in favor of resolve
    def apply(self, data: xr.Dataset) -> xr.Dataset:
        """Apply the selection to the given dataset."""
        if self.position_intervals is not None:
            parts = [
                data.isel(position=slice(start, end))
                for (start, end) in self.position_intervals
            ]
            data = xr.concat(parts, dim="position")
        return data


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


@dataclass(frozen=True)
class Genome:
    contigs: tuple[str, ...]
    offsets: tuple[int, ...]

    def __post_init__(self):
        if len(self.contigs) + 1 != len(self.offsets):
            raise ValueError(
                "Offsets must be one longer than contigs, got "
                f"{len(self.contigs)} contigs and {len(self.offsets)} offsets"
            )

    def __len__(self) -> int:
        return sum(e - s for s, e in self.contig_slices(self.contigs))

    def contig_indices(self, contigs: list[str] | None = None):
        if contigs is None:
            return np.arange(len(self.contigs))
        return np.searchsorted(self.contigs, contigs)

    def span(self, contig: str) -> tuple[int, int]:
        """Return the start and end positions for a given contig."""
        idx = self.contigs.index(contig)
        return self.offsets[idx], self.offsets[idx + 1]

    def resolve(self, sel: SpatialSelection) -> "PositionIntervals":
        """Resolve selection ranges"""
        contigs = sel.contigs or self.contigs
        spans = self.contig_slices(contigs)
        if sel.position_intervals is not None:
            # This should *reduce* the
            # Merge the position ranges
            x = PositionIntervals(sel.position_intervals).as_pyranges()
            y = PositionIntervals(spans).as_pyranges()
            spans = (
                x.intersect_overlaps(y)
                .merge_overlaps(slack=1)[["Start", "End"]]
                .to_numpy()
                .tolist()
            )
            spans = tuple(map(tuple, spans))
        return PositionIntervals(tuple((s, e) for s, e in spans if s < e))

    def as_pyranges(self, contigs: list[str] | None = None) -> pr.PyRanges:
        contig_indices = self.contig_indices(contigs)
        return pr.PyRanges(
            {
                "Start": np.array(self.offsets[:-1])[contig_indices],
                "End": np.array(self.offsets[1:])[contig_indices],
                "Chromosome": np.array(self.contigs)[contig_indices],
            }
        )

    def contig_slices(self, contigs: list[str] | None = None) -> tuple[tuple[int, int]]:
        data = self.as_pyranges(contigs)
        slices = data.merge_overlaps(slack=1)[["Start", "End"]].to_numpy().tolist()
        return tuple(map(tuple, slices))

    def resolve_intervals(self, position_intervals: PositionIntervals) -> pr.PyRanges:
        """Convert position intervals to genome space including contig information.

        Returns:
            py.PyRanges: object with contig information assigned to
                intervals in genome space (all contigs start from
                position 0)
        """
        g = self.as_pyranges()
        g["OriginalChromosome"] = g["Chromosome"].copy()
        g["Chromosome"] = "Placeholder"
        g = g.intersect_overlaps(position_intervals.as_pyranges())
        g["Chromosome"] = g["OriginalChromosome"]
        g = g.drop(["OriginalChromosome"], axis=1)
        return g


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

    def __len__(self) -> int:
        if len(self.intervals) == 0:
            return 0
        return sum([e - s for s, e in self.intervals])

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        parts = [data.isel(position=slice(s, e)) for s, e in self.intervals]
        return parts[0] if len(parts) == 1 else xr.concat(parts, dim="position")

    def as_pyranges(self) -> pr.PyRanges:
        return pr.PyRanges(
            {
                "Start": [s for s, _ in self.intervals],
                "End": [e for _, e in self.intervals],
                "Chromosome": "Placeholder",
            }
        )


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
    datastore: DataStore
    spatial: SpatialSelection = SpatialSelection()
    column: dict[str, ColumnSelection] = ()
    sample_selected: npt.NDArray[np.bool_] | None = None
    contig_selected: npt.NDArray[np.bool_] | None = None

    def __post_init__(self):
        n_samples_all, n_contigs_all = self.n_samples_all, self.n_contigs_all
        object.__setattr__(
            self,
            "sample_selected",
            np.ones(n_samples_all, dtype=bool)
            if self.sample_selected is None
            else np.asarray(self.sample_selected, dtype=bool),
        )
        object.__setattr__(
            self,
            "contig_selected",
            np.ones(n_contigs_all, dtype=bool)
            if self.contig_selected is None
            else np.asarray(self.contig_selected, dtype=bool),
        )

    def __str__(self):
        return (
            f"Coordinates summary:\n"
            f"  Samples: {self.n_samples} ({self.n_samples_all})\n"
            f"  Contigs: {self.n_contigs} ({self.n_contigs_all})\n"
            f"  Sample sets: {self.n_user_sample_sets} "
            f"({self.n_user_sample_sets_all})\n"
        )

    def __repr__(self) -> str:
        return (
            f"Coordinates(contigs={self.n_contigs}/{self.n_contigs_all}, "
            f"samples={self.n_samples}/{self.n_samples_all}, "
            f"sample_sets={self.n_user_sample_sets}/"
            f"{self.n_user_sample_sets_all}, "
            f"genome_size={self.genome_size} bp)"
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.datastore.id,
                id(self.samples_all),
                id(self.contigs_all),
                id(self.offsets),
                self.sample_selected.tobytes(),
                self.contig_selected.tobytes(),
            )
        )

    def __eq__(self, other):
        return all(
            (
                self.datastore.id == other.datastore.id,
                id(self.samples_all) == id(other.samples_all),
                id(self.contigs_all) == id(other.contigs_all),
                id(self.offsets) == id(other.offsets),
                np.array_equal(self.sample_selected, other.sample_selected),
                np.array_equal(self.contig_selected, other.contig_selected),
            )
        )

    def _replace(
        self,
        *,
        sample_selected: npt.NDArray[np.bool_] | None = None,
        contig_selected: npt.NDArray[np.bool_] | None = None,
    ) -> "Coordinates":
        """Return a new instance sharing raw data, with new selection arrays."""
        new = replace(
            self,
            sample_selected=self.sample_selected
            if sample_selected is None
            else np.asarray(sample_selected, dtype=bool),
            contig_selected=self.contig_selected
            if contig_selected is None
            else np.asarray(contig_selected, dtype=bool),
        )
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
    def with_length_filter(
        self, lower: int = 0, upper: float = np.inf
    ) -> "Coordinates":
        """Return a new Coordinates whose contigs satisfy lower <= len <= upper."""
        keep = (
            (self._contig_len_all >= lower)
            & (self._contig_len_all <= upper)
            & self.contig_selected
        )
        return self._replace(contig_selected=keep)

    def with_contigs_new(self, contigs: Iterable[str] | None = None) -> "Coordinates":
        if contigs is None:
            slices = self.contig_slices(self.contigs_all.tolist())
        else:
            unknown = set(contigs) - set(self.contigs_all.tolist())
            if unknown:
                raise ValueError(f"Unknown contigs: {sorted(unknown)}")
            slices = self.contig_slices(contigs)
        return replace(
            self,
            spatial=replace(self.spatial, position_intervals=slices, contigs=contigs),
        )

    def with_contigs(
        self,
        contigs: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed contigs. `None` resets the contig selection mask."""
        if contigs is None:
            return self._replace(contig_selected=np.ones_like(self.contig_selected))
        unknown = set(contigs) - set(self.contigs_all.tolist())
        if unknown:
            raise ValueError(f"Unknown contigs: {sorted(unknown)}")
        keep = (
            np.isin(self.contigs_all, np.asarray(list(contigs))) & self.contig_selected
        )
        return self._replace(contig_selected=keep)

    def with_all_contigs(
        self,
        contigs: Iterable[str] | None = None,
    ) -> "Coordinates":
        return self._replace(contig_selected=np.ones_like(self.contig_selected))

    def with_samples(
        self,
        samples: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed samples. `None` resets the sample selected mask."""
        if samples is None:
            return self._replace(sample_selected=np.ones_like(self.sample_selected))
        keep = (
            np.isin(self.samples_all, np.asarray(list(samples))) & self.sample_selected
        )
        return self._replace(sample_selected=keep)

    def with_sample_sets(self, sample_sets: list[str]):
        """Keep only listed sample sets. `None` resets the sample selected mask."""
        if self.default_sample_set_name in sample_sets:
            return self
        if sample_sets is None:
            return self._replace(sample_selected=np.ones_like(self.sample_selected))
        keep = (
            np.isin(self.sample_set_membership_all, sample_sets) & self.sample_selected
        )
        return self._replace(sample_selected=keep)

    def replace_contigs(self, names):
        selected_contigs = np.isin(self.contigs_all, names)
        return self._with_contig_selected(selected_contigs)

    def reset(self, *, samples: bool = True, contigs: bool = True) -> "Coordinates":
        """Return a new Coordinates with selection masks reset (nothing hidden)."""
        return self._replace(
            sample_selected=(np.ones_like(self.sample_selected) if samples else None),
            contig_selected=(np.ones_like(self.contig_selected) if contigs else None),
        )

    @property
    def base_coord(self) -> "Coordinates":
        """Return a new Coordinates with selection masks on (nothing hidden)."""
        return self._replace(
            sample_selected=np.ones_like(self.sample_selected),
            contig_selected=np.ones_like(self.contig_selected),
        )

    @property
    def samples_all(self) -> npt.NDArray:
        """All samples in the dataset"""
        return self.datastore.samples

    @property
    def n_samples_all(self) -> int:
        return self.datastore.n_samples

    @property
    def contigs_all(self) -> npt.NDArray:
        """All contigs in the dataset"""
        return self.datastore.contigs

    @property
    def n_contigs_all(self) -> int:
        return self.datastore.n_contigs

    @property
    def offsets(self) -> npt.NDArray:
        """Contig offsets"""
        return self.datastore.offsets

    @cached_property
    def _contig_len_all(self) -> npt.NDArray[np.uint32]:
        return np.diff(self.offsets).astype(np.uint32)

    @property
    def contig_len_all(self):
        return self._contig_len_all

    @property
    def contig_len(self):
        return self._contig_len_all[self.contig_selected]

    @property
    def contig_selection_is_active(self) -> bool:
        return bool(np.any(self.contig_selected))

    @property
    def contig_idx(self) -> dict:
        """Return a dict mapping of chromosome to index.
        Returns the mapping for the full set of contigs"""
        return {c: i for i, c in enumerate(self.contigs_all)}

    @cached_property
    def contig_indices(self) -> npt.NDArray[np.int64]:
        """Original-vector indices of currently selected contigs."""
        return np.flatnonzero(self.contig_selected).astype(np.int64)

    def contig_slices(self, contigs: list | None = None) -> list[np.array[int]] | None:
        """Return a list of reduced slices mapped to position coordinates."""
        if contigs is None:
            if self.contig_indices.size == 0:
                return []
            contig_indices = self.contig_indices
        else:
            contig_indices = np.searchsorted(self.contigs_all, contigs)
        slices = (
            pr.PyRanges(
                {
                    "Start": self.offsets[:-1][contig_indices],
                    "End": self.offsets[1:][contig_indices],
                    "Chromosome": "Placeholder",
                }
            )
            .merge_overlaps(slack=1)[["Start", "End"]]
            .to_numpy()
            .tolist()
        )
        return list(map(tuple, slices))

    # @property
    # def contig_selected(self):
    #     return self.contig_selected

    @property
    def contig_excluded(self):
        return ~self.contig_selected

    # @property
    # def sample_selected(self):
    #     return self.sample_selected

    @property
    def sample_excluded(self):
        return ~self.sample_selected

    # FIXME: Return self.spatial.contigs; if None, return
    # self.datastore.contigs. Need claude input on this.
    @property
    def contigs(self):
        return self.contigs_all[self.contig_selected]

    @property
    def n_contigs(self):
        return self.contigs.size

    @property
    def samples(self) -> Any:
        """Return selected samples"""
        return self.samples_all[self.sample_selected]

    @property
    def n_samples(self):
        return int(self.samples.size)

    @property
    def sample_set_membership(self):
        """Return sample sets per sample"""
        return self.datastore.sample_set_membership[self.sample_selected]

    @property
    def sample_set_membership_all(self):
        """Return sample sets per sample"""
        return self.datastore.sample_set_membership

    @property
    def default_sample_set_membership(self):
        return self.datastore.default_sample_set_membership[self.sample_selected]

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
        selected_members = self.sample_set_membership_all[self.sample_selected]
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

    @cached_property
    def genome_size(self):
        """Return the selected genome size. This is not a fixed entity"""
        return np.sum(self.contig_len, dtype=np.int64)

    @property
    def genome_size_all(self):
        """Return the total genome size, including all contigs"""
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
                "sample_set": self.default_sample_set_membership_all,
                "active": self.sample_selected,
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
                            "active": self.sample_selected,
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
                        f"{self.n_user_sample_sets}/{self.n_user_sample_sets_all}"
                    ),
                }
            ]
        )
