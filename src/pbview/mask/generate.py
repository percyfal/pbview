from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from pbview.config import SCHEMA_VERSION
from pbview.logging import cli_logger as logger
from pbview.model.datastore import DataStore
from pbview.model.thresholds import ThresholdProfile
from pbview.model.track import MissingSummaryError


def _stack(arrays: list[xr.DataArray], sample_sets: list[str]) -> xr.DataArray:
    return xr.concat(
        arrays, dim=xr.DataArray(sample_sets, dims="sample_set", name="sample_set")
    )


def _build_mask_dataset(results, base_coords, thresholds, cutoffs=None):
    """Build a dataset of boolean masks over positions.

    A mask here means a boolean array across positions. The polarity
    of a mask is inferred from the name where
      - `accessible`      : True = site is included
      - `excluded_*` : True = site is discarded based on the named
        criterion

    results: {sample_set: {"excluded_missing": arr, "excluded_coverage": arr,
                           "accessible": arr}}
    base_coords: dict of coords from the active track
    cutoffs: with cutoffs, "excluded_missing" and "accessible" are
        instead {missing cutoff: arr} and get a `missing_cutoff` dimension

    """
    sample_sets = list(results)
    ds = {}
    for layer in ["excluded_missing", "excluded_coverage", "accessible"]:
        if cutoffs is not None and layer != "excluded_coverage":
            stacked = xr.concat(
                [
                    _stack([results[s][layer][c] for s in sample_sets], sample_sets)
                    for c in cutoffs
                ],
                dim=xr.DataArray(
                    list(cutoffs), dims="missing_cutoff", name="missing_cutoff"
                ),
            ).transpose("position", "sample_set", "missing_cutoff")
        else:
            stacked = _stack([results[s][layer] for s in sample_sets], sample_sets)
            stacked = stacked.transpose("position", "sample_set")
        ds[layer] = stacked.astype(bool)

    attrs = {"schema_version": SCHEMA_VERSION, "thresholds": thresholds.to_dict()}
    if cutoffs is not None:
        attrs["missing_cutoffs"] = list(cutoffs)
    return xr.Dataset(
        data_vars=ds,
        coords={
            "offsets": base_coords["offsets"],
            "contigs": base_coords["contigs"],
            "sample_set": sample_sets,
        },
        attrs=attrs,
    )


def _save_mask(dataset: xr.Dataset, output):
    # FIXME: Optimize chunking
    chunks = {"position": 1_000_000, "sample_set": -1}
    if "missing_cutoff" in dataset.dims:
        chunks["missing_cutoff"] = -1
    dataset = dataset.chunk(chunks)
    # FIXME: add option to force (mode="w")
    dataset.to_zarr(output, mode="w-", consolidated=True)


def generate_mask(
    path: Path | str,
    *,
    threshold: Path | str,
    output: Path | str | None = None,
    track_name: str = "depth",
    progress: bool = True,
    missing_cutoffs: Sequence[int] | None = None,
) -> None:
    """Generate accessible sites for the pbzarr store.

    By default each sample set uses the missing-data cutoff in the
    threshold file. With `missing_cutoffs`, masks are generated for each
    of these cutoffs instead, and the layers that depend on the cutoff
    (`excluded_missing`, `accessible`) get a `missing_cutoff` dimension.
    """
    ds = DataStore(path=path)
    logger.info(
        "Creating accessible sites for track %s based on threshold %s",
        track_name,
        threshold,
    )
    logger.info("Writing accessible sites to pbzarr store at %s", output)
    show_progress = ProgressBar() if progress else nullcontext()

    with open(threshold, "r") as f:
        threshold_data = ThresholdProfile.from_yaml(f.read())
    track = ds.tracks[track_name]
    summaries = track.summaries
    if summaries.sum is None:
        raise MissingSummaryError(
            f"track {track_name!r} has no sum summary; run `pbview preprocess`"
        )
    cutoffs = sorted({int(c) for c in missing_cutoffs}) if missing_cutoffs else None

    results = {}
    for ss, thresholds in threshold_data.sample_sets.items():
        # Calculate excluded_coverage mask
        sums = summaries.sum.column(ss)
        excluded_coverage = (sums < thresholds.lower_coverage) | (
            sums > thresholds.upper_coverage
        )

        # Calculate excluded_missing masks, for the cutoff in the threshold
        # file or for each requested cutoff
        keys = cutoffs if cutoffs is not None else [thresholds.missing_cutoff]
        excluded_missing = {}
        accessible = {}
        for key in keys:
            miss = track.require_missingness(key).column(ss)
            excluded_missing[key] = miss > thresholds.max_missing_samples
            # Combine exclusion masks and invert for accessible sites
            accessible[key] = ~(excluded_coverage | excluded_missing[key])

        results[ss] = {
            "excluded_coverage": excluded_coverage,
            "excluded_missing": excluded_missing
            if cutoffs
            else excluded_missing[keys[0]],
            "accessible": accessible if cutoffs else accessible[keys[0]],
        }

    # Write mask to output store
    with show_progress:
        dataset = _build_mask_dataset(
            results, ds._store_coords, threshold_data, cutoffs
        )
    _save_mask(dataset, output)
