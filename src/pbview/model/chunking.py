"""
Chunking utilities for Dask arrays.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-24"

import dask.array as da

from pbview.config import MIN_POSITION_CHUNK_SIZE, TARGET_BYTES
from pbview.logging import cli_logger as logger


def position_chunk_size(
    n_columns: int,
    itemsize: int,
    target_bytes: int = TARGET_BYTES,
    min_positions: int = MIN_POSITION_CHUNK_SIZE,
) -> int:
    """Number of positions per chunk for chunks of about `target_bytes`.

    A chunk holds all `n_columns` columns (samples) of `itemsize` bytes
    each. At least `min_positions` positions are used, so a store with
    very many columns gets chunks above the target instead of tiny ones.

    Args:
        n_columns: number of columns in every chunk
        itemsize: bytes per value
        target_bytes: wanted (uncompressed) chunk size in bytes
        min_positions: smallest number of positions per chunk
    """
    if n_columns < 1 or itemsize < 1:
        raise ValueError(
            f"need at least one column and one byte per value, got "
            f"n_columns={n_columns}, itemsize={itemsize}"
        )
    return max(min_positions, target_bytes // (n_columns * itemsize))


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
