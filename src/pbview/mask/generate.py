from contextlib import nullcontext
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from pbview.config import SCHEMA_VERSION
from pbview.logging import cli_logger as logger
from pbview.model.datastore import DataStore
from pbview.model.thresholds import ThresholdProfile


def _build_mask_dataset(results, base_coords, thresholds):
    """Build mask xarray dataset.

    results: {sample_set: {"missingness": arr, "coverage": arr, "combined": arr}}
    base_coords: dict of coords from the active track
    """
    sample_sets = list(results)
    layers = ["missingness", "coverage", "combined"]

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


def generate_masks(
    path: Path | str,
    *,
    threshold: Path | str,
    output: Path | str | None = None,
    track_name: str = "depth",
    progress: bool = True,
    sampleinfo: Path | str | None = None,
) -> None:
    """Generate masks for the pbzarr store."""
    ds = DataStore(path=path, sampleinfo=sampleinfo)
    logger.info(
        "Creating mask for track %s based on threshold %s", track_name, threshold
    )
    logger.info("Writing mask to pbzarr store at %s", output)
    show_progress = ProgressBar() if progress else nullcontext()

    track = ds.tracks[track_name]
    with open(threshold, "r") as f:
        threshold_data = ThresholdProfile.from_yaml(f.read())
    sums = track._pre

    results = {}
    for ss, thresholds in threshold_data.sample_sets.items():
        # Calculate coverage mask
        sums = track._pre.sum["values"].sel(sample_set=ss)
        coverage_mask = (sums < thresholds.lower_coverage) | (
            sums > thresholds.upper_coverage
        )

        # Calculate missingness mask
        key = thresholds.missing_cutoff
        miss = track._pre.missing_cutoff[key]["values"].sel(sample_set=ss)
        missingness_mask = miss > thresholds.max_missing_samples

        # Combine masks
        combined_mask = coverage_mask | missingness_mask

        results[ss] = {
            "coverage": coverage_mask,
            "missingness": missingness_mask,
            "combined": combined_mask,
        }

    # Write mask to output store
    with show_progress:
        dataset = _build_mask_dataset(results, ds._store_coords, threshold_data)
    _save_mask(dataset, output)
