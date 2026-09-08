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
        self._samples: npt.NDArray = np.asarray(samples)
        self._contigs: npt.NDArray = np.asarray(contigs)
        self._offsets: npt.NDArray[np.int64] = np.asarray(offsets, dtype=np.int64)
        # Masks: True = hidden. Default = keep everything
        n_s, n_c = self._samples.size, self._contigs.size
        self.sample_mask: npt.NDArray[np.bool_] = (
            np.zeros(n_s, dtype=bool)
            if sample_mask is None
            else np.asarray(sample_mask, dtype=bool)
        )
        self.contig_mask: npt.NDArray[np.bool_] = (
            np.zeros(n_c, dtype=bool)
            if contig_mask is None
            else np.asarray(contig_mask, dtype=bool)
        )
        self._default_sample_sets: npt.NDArray = np.repeat(
            config.DEFAULT_SAMPLE_SET, self._samples.size
        )
        self._sample_sets: npt.NDArray = np.asarray(
            self._default_sample_sets if sample_sets is None else sample_sets
        )
        # Freeze the underlying arrays to catch accidental mutation.
        for arr in (
            self._samples,
            self._contigs,
            self._offsets,
            self.sample_mask,
            self.contig_mask,
            self._default_sample_sets,
            self._sample_sets,
        ):
            arr.setflags(write=False)

    def __str__(self):
        return (
            f"Coordinates summary:\n"
            f"  Samples: {len(self.samples)} ({len(self._samples)})\n"
            f"  Contigs: {len(self.contigs)} ({len(self._contigs)})\n"
            f"  Sample sets: {len(set(list(self.sample_sets)))} "
            f"({len(set(list(self._sample_sets)))})\n"
        )

    def __repr__(self) -> str:
        return (
            f"Coordinates(active_contigs={self.contigs.size}/{self._contigs.size}, "
            f"active_samples={self.samples.size}/{self._samples.size}, "
            f"size={self.size:_} bp)"
        )

    def __hash__(self) -> int:
        return hash(
            (
                id(self._samples),
                id(self._contigs),
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
        new._samples = self._samples
        new._contigs = self._contigs
        new._offsets = self._offsets
        new._default_sample_sets = self._default_sample_sets
        new._sample_sets = self._sample_sets
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
        return new

    # Factory methods
    def with_length_filter(
        self, lower: int = 0, upper: float = np.inf
    ) -> "Coordinates":
        """Return a new Coordinates whose contigs satisfy lower ≤ len ≤ upper."""
        keep = (self._contig_len_all >= lower) & (self._contig_len_all <= upper)
        return self._replace(contig_mask=~keep)

    def with_contigs(
        self,
        contigs: Iterable[str] | None = None,
        lower: int = 0,
        upper: float = np.inf,
    ) -> "Coordinates":
        """Keep only listed contigs and contigs whose lengths satisfy
        lower ≤ len ≤ upper. `None` resets the contig mask."""
        if (
            ((contigs is None) or (len(contigs) == 0))
            and (lower == 0)
            and (np.isinf(upper))
        ):
            return self._replace(contig_mask=np.zeros_like(self.contig_mask))
        keep = ~self.with_length_filter(lower=lower, upper=upper).contig_mask
        if contigs is not None:
            keep = keep & np.isin(self._contigs, np.asarray(list(contigs)))
        return self._replace(contig_mask=~keep)

    def with_samples(
        self,
        samples: Iterable[str] | None = None,
        sample_sets: Iterable[str] | None = None,
    ) -> "Coordinates":
        """Keep only listed samples. `None` resets the sample mask."""
        if (samples is None) and (sample_sets is None):
            return self._replace(sample_mask=np.zeros_like(self.sample_mask))
        if sample_sets is not None:
            keep1 = np.isin(self._sample_sets, sample_sets)
        else:
            keep1 = np.ones(self._samples.size, dtype=bool)
        if samples is not None:
            keep2 = np.isin(self._samples, np.asarray(list(samples)))
        else:
            keep2 = np.ones(self._samples.size, dtype=bool)
        keep = keep1 & keep2
        return self._replace(sample_mask=~keep)

    def reset(self, *, samples: bool = True, contigs: bool = True) -> "Coordinates":
        """Return a new Coordinates with masks reset (nothing hidden)."""
        return self._replace(
            sample_mask=(np.zeros_like(self.sample_mask) if samples else None),
            contig_mask=(np.zeros_like(self.contig_mask) if contigs else None),
        )

    @cached_property
    def _contig_len_all(self) -> npt.NDArray[np.uint32]:
        return np.diff(self._offsets).astype(np.uint32)

    @property
    def contig_mask_is_active(self) -> bool:
        return bool(np.any(self.contig_mask))

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
    def contig_len(self):
        return self._contig_len_all[~self.contig_mask]

    @property
    def contig_idx(self) -> dict:
        """Return a dict mapping of chromosome to index.
        Returns the mapping for the full set of contigs"""
        return {c: i for i, c in enumerate(self._contigs)}

    @cached_property
    def active_contig_indices(self) -> npt.NDArray[np.int64]:
        """Original-vector indices of currently active contigs."""
        return np.flatnonzero(~self.contig_mask).astype(np.int64)

    def contig_slices(self) -> list[tuple(int, int)] | None:
        """Return a list of reduced slices for subsetting data by position"""
        idx = self.active_contig_indices
        if idx.size == 0:
            return []

        starts = self._offsets[idx]
        ends = self._offsets[idx + 1]

        merged: list[tuple[int, int]] = [(int(starts[0]), int(ends[0]))]
        for s, e in zip(starts[1:], ends[1:]):
            prev_s, prev_e = merged[-1]
            if s == prev_e:
                merged[-1] = (prev_s, int(e))
            else:
                merged.append((int(s), int(e)))
        return merged

    @property
    def contigs(self):
        return self._contigs[~self.contig_mask]

    @property
    def contigs_size(self):
        return int(self.contigs.size)

    @property
    def samples(self):
        return self._samples[~self.sample_mask]

    @property
    def samples_size(self):
        return int(self.samples.size)

    @property
    def sample_sets(self):
        """Return sample sets"""
        return self._sample_sets[~self.sample_mask]

    @property
    def sample_sets_size(self):
        """Return sample sets size"""
        return int(len(set(self.sample_sets)))

    @property
    def default_sample_sets(self):
        """Return default sample sets"""
        return (self._default_sample_sets[~self.sample_mask],)

    @cached_property
    def genome_size(self):
        return np.sum(self._contig_len_all, dtype=np.int64)

    @property
    def size(self):
        return np.sum(self.contig_len, dtype=np.int64)

    # def mask_contigs(self, lower: int = 0, upper: int | float = np.inf):
    #     """Mask contigs by length"""
    #     self.lower = lower
    #     self.upper = upper
    #     self.contig_mask = ~np.array(
    #         [(x >= lower) & (x <= upper) for x in self._contig_len]
    #     )

    # def mask_samples(self, samples=None) -> None:
    #     """Mask samples in argument. Reset mask if no samples"""
    #     if samples is None:
    #         self.sample_mask: list[bool] = np.zeros(len(self._samples), dtype=bool)
    #     else:
    #         self.sample_mask = np.isin(self._samples, samples)

    def _sample_sets_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "sample set": self._default_sample_sets,
                "Active": ~self.sample_mask,
                "Total": self._samples,
            }
        )
        if not np.all(self._sample_sets == config.DEFAULT_SAMPLE_SET):
            df = pd.concat(
                df,
                pd.DataFrame(
                    {
                        "sample_sets": self._sample_sets,
                        "Active": ~self.sample_mask,
                        "Total": self._samples,
                    }
                ),
            )
        return df.groupby("sample set").agg({"Active": "sum", "Total": "count"})

    def _coordinates_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [len(self.contigs), len(self._contigs)],
                [len(self.samples), len(self._samples)],
            ],
            columns=["Active", "Total"],
            index=["Contigs", "Samples"],
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame summary of active samples, contigs and sample sets."""
        return pd.concat(
            [self._coordinates_dataframe(), self._sample_sets_dataframe()],
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
        if coord.contigs_size == 0:
            return self._data.sel(position=[])
        parts = [res.isel(position=slice(a, b)) for a, b in coord.contig_slices()]

        return xr.concat(parts, dim="position") if len(parts) > 1 else parts[0]

    def __repr__(self) -> str:
        return f"<Track(name={self.name}, data={self.data()})>"

    def __str__(self) -> str:
        return f"Track(name={self.name})"

    def hist(
        self,
        bins: npt.NDArray[int],
        coord: Coordinates,
        *,
        threshold: int | None = None,
    ):
        """Calculate histogram for a given number of bins and coordinate state.
        If threshold is set calculate a thresholded histogram.

        The histogram values are calculated as the sum of values across all samples.

        Args:
            bins: number of bins for histogram
            coord: coordinate state for data selection
            threshold: optional threshold value for thresholded histogram
        Returns:
            Histogram as a numpy array
        """
        if threshold is not None:
            logger.info("Calculating thresholded histogram for '%s' track", self.name)
            return self._hist_threshold_cached(tuple(bins.tolist()), coord, threshold)
        logger.info("Calculating histogram of coverage for '%s' track", self.name)
        return self._hist_cached(tuple(bins.tolist()), coord)

    @lru_cache(maxsize=16)
    def _hist_cached(self, bins: npt.NDArray[int], coord: Coordinates):
        bins = np.asarray(bins)
        with ProgressBar():
            hist, _ = da.histogram(self.data(coord).sum("sample")["values"], bins=bins)
        return hist.compute()

    @lru_cache(maxsize=16)
    def _hist_threshold_cached(
        self, bins: npt.NDArray, coord: Coordinates, threshold: int
    ):
        bins = np.asarray(bins)
        with ProgressBar():
            hist, _ = da.histogram(
                (self.data(coord).sum("sample")["values"] > threshold).astype(
                    dtype=np.uint8
                ),
                bins=bins,
            )
        return hist.compute()

    # FIXME: move? This is really a Coordinates + SelectionState
    # summary; the only track-related information here is the track
    # name
    def summary(self, coord: Coordinates, lower: int = 0, upper: float = np.inf):
        """Return a track summary of coordinate selection relative to
        base coordinates."""
        df = pd.DataFrame(
            [
                {
                    "type": f"{str(self)} Summary",
                    "track": self.name,
                    "genome_size": coord.genome_size,
                    "selected_size": coord.size,
                    "selected_size (%)": np.round(
                        coord.size / coord.genome_size * 100.0, 2
                    ),
                    "n_samples": coord.samples_size,
                    "n_contigs": len(coord._contigs),
                    "min_contig_len": lower,
                    "max_contig_len": upper,
                    "n_contigs_selected": coord.contigs_size,
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
