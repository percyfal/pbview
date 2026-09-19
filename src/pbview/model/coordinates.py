"""
Coordinates class and helper functions.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyranges1 as pr
import xarray as xr

if TYPE_CHECKING:
    from .datastore import DataStore

from pbview import config

xr.set_options(display_expand_attrs=False)


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
        datastore: DataStore,
        *,
        contigs: npt.ArrayLike | xr.DataArray,
        offsets: npt.ArrayLike | xr.DataArray,
        sample_sets: npt.ArrayLike | None = None,
        sample_mask: npt.NDArray[np.bool_] | None = None,
        contig_mask: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        self._datastore = datastore
        self._contigs_all: npt.NDArray = np.asarray(contigs)
        self._offsets: npt.NDArray[np.int64] = np.asarray(offsets, dtype=np.int64)
        # Masks: True = hidden. Default = keep everything
        n_samples_all, n_contigs_all = self.n_samples_all, self._contigs_all.size
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
        self._sample_set_membership: npt.NDArray = np.asarray(
            self._datastore.default_sample_set_membership
            if sample_sets is None
            else sample_sets
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
            self._contigs_all,
            self._offsets,
            self.sample_mask,
            self.contig_mask,
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
                id(self.samples_all),
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

        new._datastore = self._datastore
        new._contigs_all = self._contigs_all
        new._offsets = self._offsets
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
                members = self.samples_all
            else:
                members = self.samples_all[self._sample_set_membership == name]
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

    def with_all_contigs(
        self,
        contigs: Iterable[str] | None = None,
    ) -> "Coordinates":
        return self._replace(contig_mask=np.zeros_like(self.contig_mask))

    def with_samples(
        self,
        samples: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed samples. `None` resets the sample mask."""
        if samples is None:
            return self._replace(sample_mask=np.zeros_like(self.sample_mask))
        keep = np.isin(self.samples_all, np.asarray(list(samples))) & ~self.sample_mask
        return self._replace(sample_mask=~keep)

    def with_sample_sets(self, sample_sets: list[str]):
        """Keep only listed sample sets. `None` resets the sample mask."""
        if self.default_sample_set_name in sample_sets:
            return self
        if sample_sets is None:
            return self._replace(sample_mask=np.zeros_like(self.sample_mask))
        keep = np.isin(self._sample_set_membership, sample_sets) & ~self.sample_mask
        return self._replace(sample_mask=~keep)

    def replace_contigs(self, names):
        mask = np.isin(self.contigs_all, names)
        return self._with_contig_mask(mask)

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

    @property
    def samples_all(self) -> npt.NDArray:
        """All samples in the dataset"""
        return self._datastore.samples

    @property
    def n_samples_all(self) -> int:
        return self._datastore.n_samples

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
    def contigs_all(self):
        return self._contigs_all

    @property
    def n_contigs(self):
        return self.contigs.size

    @property
    def n_contigs_all(self) -> int:
        return int(self._contigs_all.size)

    @property
    def samples(self) -> Any:
        return self.samples_all[~self.sample_mask]

    @property
    def n_samples(self):
        return int(self.samples.size)

    @property
    def sample_set_membership(self):
        """Return sample sets per sample"""
        return self._sample_set_membership[~self.sample_mask]

    @property
    def default_sample_set_membership(self):
        return self._datastore.default_sample_set_membership[~self.sample_mask]

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
                "sample_set": self._datastore.default_sample_set_membership,
                "active": ~self.sample_mask,
                "sample": self.samples_all,
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
                        f"{self.n_user_sample_sets}"  # /{self.n_user_sample_sets_all}"
                    ),
                }
            ]
        )
