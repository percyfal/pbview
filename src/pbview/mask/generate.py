from contextlib import nullcontext
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from pbview.config import SCHEMA_VERSION
from pbview.logging import cli_logger as logger
from pbview.model.datastore import DataStore
from pbview.model.thresholds import ThresholdProfile


def _build_mask_dataset(results, base_coords, thresholds):
    """Build a dataset of boolean masks over positions.

    A mask here means a boolean array across positions. The polarity
    of a mask is inferred from the name where
      - `accessible`      : True = site is included
      - `excluded_*` : True = site is discarded based on the named
        criterion

    results: {sample_set: {"excluded_missing": arr, "excluded_coverage": arr,
                           "accessible": arr}}
    base_coords: dict of coords from the active track

    """
    sample_sets = list(results)
    layers = ["excluded_missing", "excluded_coverage", "accessible"]

    ds = {}
    for layer in layers:
        stacked = xr.concat(
            [results[s][layer] for s in sample_sets],
            dim=xr.DataArray(sample_sets, dims="sample_set", name="sample_set"),
        ).transpose("position", "sample_set")
        ds[layer] = stacked.astype(bool)

    dataset = xr.Dataset(
        data_vars=ds,
        coords={
            "offsets": base_coords["offsets"],
            "contigs": base_coords["contigs"],
            "sample_set": sample_sets,
        },
        attrs={
            "schema_version": SCHEMA_VERSION,
            "thresholds": thresholds.to_dict(),
        },
    )
    return dataset


def _save_mask(dataset: xr.Dataset, output):
    # FIXME: Optimize chunking
    dataset = dataset.chunk({"position": 1_000_000, "sample_set": -1})
    # FIXME: add option to force (mode="w")
    dataset.to_zarr(output, mode="w-", consolidated=True)


def generate_mask(
    path: Path | str,
    *,
    threshold: Path | str,
    output: Path | str | None = None,
    track_name: str = "depth",
    progress: bool = True,
    sampleinfo: Path | str | None = None,
) -> None:
    """Generate accessible sites for the pbzarr store."""
    ds = DataStore(path=path, sampleinfo=sampleinfo)
    logger.info(
        "Creating accessible sites for track %s based on threshold %s",
        track_name,
        threshold,
    )
    logger.info("Writing accessible sites to pbzarr store at %s", output)
    show_progress = ProgressBar() if progress else nullcontext()

    track = ds.tracks[track_name]
    with open(threshold, "r") as f:
        threshold_data = ThresholdProfile.from_yaml(f.read())
    sums = track._pre

    results = {}
    for ss, thresholds in threshold_data.sample_sets.items():
        # Calculate excluded_coverage mask
        sums = track._pre.sum["values"].sel(sample_set=ss)
        excluded_coverage = (sums < thresholds.lower_coverage) | (
            sums > thresholds.upper_coverage
        )

        # Calculate excluded_missing mask
        key = thresholds.missing_cutoff
        miss = track._pre.missing_cutoff[key]["values"].sel(sample_set=ss)
        excluded_missing = miss > thresholds.max_missing_samples

        # Combine exclusion masks and invert for accessible sites
        accessible = ~(excluded_coverage | excluded_missing)

        results[ss] = {
            "excluded_coverage": excluded_coverage,
            "excluded_missing": excluded_missing,
            "accessible": accessible,
        }

    # Write mask to output store
    with show_progress:
        dataset = _build_mask_dataset(results, ds._store_coords, threshold_data)
    _save_mask(dataset, output)
