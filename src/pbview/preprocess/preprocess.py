from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from pbview.logging import app_logger as logger
from pbview.model.datastore import DataStore
from pbview.model.histogram import compute_threshold_defaults, hist_stats

from .track import compute_track_missingness, compute_track_sum


def preprocess(
    path: Path | str,
    track: str = "depth",
    progress: bool = True,
    chunk_size: int = 1_000_000,
    sampleinfo: Path | str | None = None,
    missing_cutoff: int = 3,
    preprocess_sum: bool = True,
    preprocess_missingness: bool = True,
) -> None:
    """Preprocess the pbzarr store for faster access.

    Add track_sum and / or track_count tracks for all samplesets.
    """
    show_progress = ProgressBar() if progress else nullcontext()

    ds = DataStore(path=path, sampleinfo=sampleinfo)

    if preprocess_sum:
        ds_sum = compute_track_sum(track=ds.tracks[track], base_coord=ds.base_coord)
        logger.info("Writing track sum to pbzarr store at %s", path)
        with show_progress:
            ds_sum.to_zarr(path, group=f"{track}_sum", mode="w")

    if preprocess_missingness:
        ds_miss = compute_track_missingness(
            track=ds.tracks[track],
            base_coord=ds.base_coord,
            missing_cutoff=missing_cutoff,
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
    sampleinfo: Path | str | None = None,
    missing_cutoff: int = 3,
) -> None:
    """Compute default thresholds for sum data"""
    show_progress = ProgressBar() if progress else nullcontext()
    ds = DataStore(path=path, sampleinfo=sampleinfo)

    defaults = compute_threshold_defaults(ds, track_name)

    track = ds.tracks[track_name]
    coord = ds.base_coord
    # Loop sample sets and for each sum data calculate histogram using
    # the defaults for maxbins
    rows = []
    with show_progress:
        for ss in ds.sample_set_names:
            # FIXME: Add option to choose by sample stats
            by_sample = False
            n = coord.with_sample_sets([ss]).n_samples if by_sample else 1
            maxbins = defaults[f"maxbins_{ss}"]
            bins = np.arange(maxbins + 1)

            counts, bins = track.coverage_hist(
                bins=bins, coord=coord.with_sample_sets([ss])
            )
            lo, up = defaults[f"lower_coverage_{ss}"], defaults[f"upper_coverage_{ss}"]
            stats = hist_stats(counts, bins)
            mask = (bins >= lo / n) & (bins <= up / n)
            accessible = int(counts[mask].sum())
            rows.append(
                {
                    "sample_set": ss,
                    "lower": lo,
                    "lower_by_sample": lo / coord.with_sample_sets([ss]).n_samples,
                    "upper": up,
                    "upper_by_sample": up / coord.with_sample_sets([ss]).n_samples,
                    "accessible_bp": accessible,
                    "frac_active_genome": accessible / coord.genome_size,
                    "frac_full_genome": accessible / coord.genome_size_all,
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
