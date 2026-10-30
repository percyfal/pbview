"""
Preprocessing track module.

Preprocess pbzarr track for faster access.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-15"

import numpy as np
import xarray as xr

from pbview.logging import app_logger as logger
from pbview.model.grouping import grouping_hash
from pbview.model.track import Track

_INHERITED_ATTRS = (
    "zarr_conventions",
    "perbase:version",
    "perbase:coordinates",
    "perbase:genome_checksum",
)


def _summarize(
    track: Track, membership: xr.DataArray, values: xr.DataArray, provenance: str
) -> xr.Dataset:
    """Sum `values` over the columns of each group in `membership`."""
    column_dim, group_dim = membership.dims
    values, member = xr.align(values, membership.astype(np.int32), join="exact")
    data = (
        xr.dot(values, member, dim=column_dim)
        .astype(np.int32)
        .transpose("position", group_dim)
    )
    attrs = {k: v for k, v in track.attrs.items() if k in _INHERITED_ATTRS}
    return xr.Dataset(
        data_vars={"values": data.chunk({group_dim: -1, "position": 1_000_000})},
        coords={
            group_dim: membership[group_dim].values,
            "contigs": ("contig", track.coords["contigs"].values),
            "offsets": ("contig_boundary", track.coords["offsets"].values),
        },
        attrs={
            **attrs,
            "perbase:provenance": provenance,
            "perbase:column_dim": group_dim,
            "perbase:source_track": track.name,
            "perbase:source_grouping_hash": grouping_hash(membership),
        },
    )


def compute_track_sum(track: Track, membership: xr.DataArray) -> xr.Dataset:
    """Sum a track over the groups of a (column, group) membership.

    Args:
        track: The input track data.
        membership: Boolean membership of the track columns in groups.
    Returns:
        Dataset with `values` of dims (position, group dim), one column per group.
    """
    logger.info("Computing sums for %d groups", membership.shape[1])
    return _summarize(
        track, membership, track.values.astype(np.int32), "preprocess_sum"
    )


def compute_track_missingness(
    track: Track, membership: xr.DataArray, missing_cutoff: int = 3
) -> xr.Dataset:
    """Count missing columns (value <= missing_cutoff) per group and position."""
    logger.info("Computing missingness for %d groups", membership.shape[1])
    return _summarize(
        track,
        membership,
        (track.values <= missing_cutoff).astype(np.int32),
        "preprocess_missingness",
    )
