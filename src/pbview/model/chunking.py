"""
Chunking utilities for Dask arrays.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-24"

import dask.array as da

from pbview.config import TARGET_BYTES
from pbview.logging import cli_logger as logger


# FIXME: expose TARGET_BYTES via CLI or environment variable
# FIXME: add reduce_axis to reduce along a specific axis
# FIXME: add specific function for chunking along samples vs position
def optimal_chunks_dask(
    arr: da.Array, reduce_along: str, target_bytes: int = TARGET_BYTES
) -> dict:
    """Return optimal chunk sizes for a dask array based on target bytes."""
    logger.debug("Calculating optimal chunks for target bytes: %s", target_bytes)
    itemsize = arr.dtype.itemsize
    reduce_len = arr.sizes[reduce_along]
    other_dim = next(d for d in arr.dims if d != reduce_along)
    other_chunk = max(1, target_bytes // (reduce_len * itemsize))
    other_chunk = min(other_chunk, arr.sizes[other_dim])
    return {
        reduce_along: -1,
        other_dim: int(other_chunk),
    }
