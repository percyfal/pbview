from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from pbview.logging import app_logger as logger
from pbview.model.datastore import DataStore
from pbview.model.grouping import (
    IDENTITY_DIM,
    read_sampleset_info,
    write_column_groupings,
)
from pbview.model.histogram import compute_threshold_defaults, hist_stats
from pbview.model.track import Track

from .track import compute_track_missingness, compute_track_sum


def _source_membership(
    path: Path | str, track: Track, sampleset_info: Path | str | None
) -> xr.DataArray:
    """Grouping that the summaries are computed from.

    A supplied sampleset_info table is written to the track (keeping its
    other groupings); otherwise the track's single stored grouping is used,
    falling back to the identity.
    """
    if sampleset_info is not None:
        membership = read_sampleset_info(sampleset_info, track.columns)
        keep = [
            g
            for dim, g in track.groupings.items()
            if dim not in (IDENTITY_DIM, membership.dims[1])
        ]
        write_column_groupings(path, track.name, [*keep, membership])
        return membership
    stored = [g for dim, g in track.groupings.items() if dim != IDENTITY_DIM]
    if len(stored) > 1:
        raise ValueError(
            f"track {track.name!r} has several groupings "
            f"{[g.dims[1] for g in stored]}; "
            "pass --sampleset-info to choose one"
        )
    return stored[0] if stored else track.groupings[IDENTITY_DIM]


def preprocess(
    path: Path | str,
    track: str = "depth",
    progress: bool = True,
    chunk_size: int = 1_000_000,
    sampleset_info: Path | str | None = None,
    missing_cutoff: int = 3,
    preprocess_sum: bool = True,
    preprocess_missingness: bool = True,
) -> None:
    """Preprocess the pbzarr store for faster access.

    Add track_sum and / or track_count tracks for all samplesets.
    """
    show_progress = ProgressBar() if progress else nullcontext()

    ds = DataStore(path=path)
    source = ds.tracks[track]
    membership = _source_membership(path, source, sampleset_info)

    if preprocess_sum:
        ds_sum = compute_track_sum(source, membership)
        logger.info("Writing track sum to pbzarr store at %s", path)
        with show_progress:
            ds_sum.to_zarr(path, group=f"{track}_sum", mode="w")

    if preprocess_missingness:
        ds_miss = compute_track_missingness(
            source, membership, missing_cutoff=missing_cutoff
        )
        logger.info("Writing track missingness to pbzarr store at %s", path)
        with show_progress:
            ds_miss.to_zarr(
                path,
                group=f"{track}_missing_cutoff={missing_cutoff}",
                mode="w",
            )


def compute_thresholds(
    path: Path | str,
    track_name: str = "depth",
    progress: bool = True,
    chunk_size: int = 1_000_000,
    missing_cutoff: int = 3,
) -> None:
    """Compute default thresholds for sum data"""
    show_progress = ProgressBar() if progress else nullcontext()
    ds = DataStore(path=path)

    defaults = compute_threshold_defaults(ds, track_name)

    track = ds.tracks[track_name]
    coord = ds.base_coord
    # Loop sample sets and for each sum data calculate histogram using
    # the defaults for maxbins
    rows = []
    with show_progress:
        for ss in track.groups("sample_set"):
            # FIXME: Add option to choose by sample stats
            by_sample = False
            n_samples = track.n_selected(coord.with_sample_sets([ss]).column)
            n = n_samples if by_sample else 1
            maxbins = defaults[f"maxbins_{ss}"]
            bins = np.arange(maxbins + 1)

            counts, bins = track.coverage_hist(
                bins=bins, coord=coord.with_sample_sets([ss])
            )
            lo, up = defaults[f"lower_coverage_{ss}"], defaults[f"upper_coverage_{ss}"]
            stats = hist_stats(counts, bins)
            mask = (bins >= lo / n) & (bins <= up / n)
            accessible = int(counts[mask].sum())
            genome = coord.datastore.genome  # SHIM: remove with Coordinates.datastore
            rows.append(
                {
                    "sample_set": ss,
                    "lower": lo,
                    "lower_by_sample": lo / n_samples,
                    "upper": up,
                    "upper_by_sample": up / n_samples,
                    "accessible_bp": accessible,
                    "frac_active_genome": accessible / genome.size,
                    "frac_full_genome": accessible / genome.size,
                    "median": stats["median"],
                    "mean": stats["mean"],
                    "60%": stats["mean"] * 0.6,
                    "70%": stats["mean"] * 0.7,
                    "80%": stats["mean"] * 0.8,
                    "90%": stats["mean"] * 0.9,
                    "med + 1std": stats["median"] + stats["std"],
                    "med + 2std": stats["median"] + 2 * stats["std"],
                    "std": stats["std"],
                }
            )
        df = pd.DataFrame(rows)

    print(df)
