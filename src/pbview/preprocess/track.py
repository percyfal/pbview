"""
Preprocessing track module.

Preprocess pbzarr track for faster access.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-15"

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from pbview.datastore import Coordinates, Track
from pbview.logging import app_logger as logger


def compute_track_sum(track: Track, base_coord: Coordinates) -> xr.Dataset:
    """
    Compute summary statistics for a given track.

    Parameters
    ----------
    track : Track
        The input track data.
    base_coord : Coordinates
        Base coordinate selection.

    Returns
    -------
    xr.Dataset
        A dataset containing the computed sums.
    """
    logger.info(f"Computing sums for {base_coord.n_sample_sets} sample sets")
    with ProgressBar():
        data = xr.concat(
            [
                track.data(base_coord.with_sample_sets([name]))["values"]
                .sum("sample")
                .astype(np.int32)
                for name in base_coord.sample_set_names
            ],
            dim=xr.DataArray(
                base_coord.sample_set_names, dims="sample_set", name="sample_set"
            ),
        )
    data = data.transpose("position", "sample_set")
    # Coordinates are for full dataset
    ds_out = xr.Dataset(
        data_vars={"values": data.chunk({"sample_set": -1, "position": 1_000_000})},
        coords={
            "offsets": base_coord._offsets,
            "contigs": base_coord._contigs_all,
            "sample_set": base_coord.sample_set_names,
        },
    )
    return ds_out
