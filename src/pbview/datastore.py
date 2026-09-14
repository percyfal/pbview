"""
Datastore model
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-01"

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


def import_d4(
    path: Path | str,
    d4: list[Path] | list[str],
    *,
    track: str = "depth",
    workers: int = 1,
    progress: bool = True,
) -> None:
    """Import d4 files to pbzarr store.

    This function will create a new pbzarr store at `path` and add
    tracks by importing data from the specified sources.
    """
    # Create the pbzarr store
    if Path(path).exists():
        logger.debug("Path %s already exists, not creating store", path)
    else:
        logger.info("Creating pbzarr store at %s", path)
        pbzarr.create_store(path)

    sources: list[tuple[str, str]] = []
    for fn in d4:
        if not re.search(r".d4$", fn):
            raise ValueError(f"Input file {fn} does not have .d4 extension")
        fn = Path(fn)
        sample = re.sub(".per-base.d4", "", fn.name)
        sources.append((str(fn), sample))

    # Import data into the store
    logger.info("Importing %i data sources", len(sources))
    try:
        report = pbzarr.import_d4(
            destination=str(path),
            track=track,
            sources=sources,
            workers=workers,
            progress=progress,
            chunk_size=1_000_000,
            column_dim="sample",
        )
    except pbzarr.PbzError as e:
        logger.error("Error importing d4 files: %s", e)
        raise
    except Exception as e:
        logger.error("Error importing d4 files: %s", e)
        logger.error(report)
        raise


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
            f"selected_size={self.selected_size} bp)"
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
        new.user_sample_set_names = np.asarray(
            sorted(list(set(new.sample_set_membership)))
        )
        new.sample_set_names = self.sample_set_names  # FIXME: is this wrong?
        return new

    # Factory methods
    def with_length_filter(
        self, lower: int = 0, upper: float = np.inf
    ) -> "Coordinates":
        """Return a new Coordinates whose contigs satisfy lower ≤ len ≤ upper."""
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
    def n_contigs_all(self):
        return int(self._contigs_all.size)

    @property
    def samples(self):
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
        return self.n_user_sample_sets > 1

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

    def _sample_set_membership_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "sample set": self._default_sample_set_membership,
                "Active": ~self.sample_mask,
                "Total": self._samples_all,
            }
        )
        if not self.has_sample_sets:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "sample set": self._sample_set_membership,
                            "Active": ~self.sample_mask,
                            "Total": self._samples,
                        }
                    ),
                ]
            )
        return df.groupby("sample set").agg({"Active": "sum", "Total": "count"})

    def _coordinates_dataframe(self) -> pd.DataFrame:
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
            [self._coordinates_dataframe(), self._sample_set_membership_dataframe()],
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


class Track:
    """A representation of a pbzarr data track.

    Args:
        track_name: name of track
        data: pbzarr data group

    Returns:
        A Track object.
    """

    def __init__(self, track_name: str, data: xr.DataTree):
        self.name = track_name
        self._data = data

    def data(self, coord: Coordinates) -> xr.DataArray | xr.Dataset:
        """Return track data based on `Coordinates` state."""
        res = self._data.sel(sample=coord.samples)
        if not coord.contig_mask_is_active:
            return res
        if coord.n_contigs == 0:
            return self._data.sel(position=[])
        parts = [res.isel(position=slice(a, b)) for a, b in coord.contig_slices()]

        return xr.concat(parts, dim="position") if len(parts) > 1 else parts[0]

    def __repr__(self) -> str:
        return f"<Track(name={self.name})>"

    def __str__(self) -> str:
        return f"Track(name={self.name})"

    @lru_cache(maxsize=64)
    def _coverage_hist_cached(self, bins: npt.NDArray[int], coord: Coordinates):
        logger.info("Calculating histogram of coverage for '%s' track", self.name)
        logger.debug("Current selection: %s", coord)
        bins = np.asarray(bins)
        with ProgressBar():
            hist, bins = da.histogram(
                self.data(coord).sum("sample")["values"], bins=bins
            )
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

        with ProgressBar():
            hist, bins = da.histogram(
                da.sum(
                    (self.data(coord)["values"].values > threshold).astype(np.int64),
                    axis=1,
                ),
                bins=bins,
            )
        return hist.compute(), bins

    def missingness_hist(
        self,
        bins: npt.NDArray[int],
        coord: Coordinates,
        threshold: int = 0,
    ):
        """Calculate missingness histogram for a given number of bins
        and coordinate state.

        The histogram values are calculated as the number of values across all
        samples greater than threshold.

        Args:
            bins: number of bins for histogram
            coord: coordinate state for data selection
            threshold: threshold value to treat values as missing (default: 0)
        Returns:
            Histogram, bins as a numpy arrays
        """
        return self._missingness_hist_cached(tuple(bins.tolist()), coord, threshold)

    # FIXME: would it be possible to cheaply calculate the *combined*
    # effect of coverage and missing data filters? This would require
    # applying selected thresholds to *all* positions jointly and
    # returning the .dims["position"] attribute. Modifying the
    # thresholds for the individual histograms is cheap; the joint
    # operation is not. Would need reactive "UPDATE" button.
    def size(self):
        pass

    # FIXME: the size is the size of the filtered coordinates to
    # indicate the "active genome": the variable name should reflect
    # this
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
        sample_sets = (
            pd.read_table(sampleinfo, header=None, sep=r"\s+", index_col=0)[1].values
            if sampleinfo is not None
            else None
        )
        self.base_coord = Coordinates.from_datatree(
            self.store[track], sample_sets=sample_sets
        )
        self.tracks = {
            name: Track(
                track_name=name,
                data=self.store[name],
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
