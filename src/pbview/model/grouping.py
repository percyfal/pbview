"""
Groupings of track columns.

A grouping is a boolean (column, group) membership `DataArray`, for example
`(sample, sample_set)`; a column may belong to several groups.

- Every track exposes a derived *identity* grouping `(column_dim, "identity")`
  with the single set `ALL` (`IDENTITY_DIM`, `config.DEFAULT_GROUPING_SET`).
  It is computed from the column labels and never stored. A track whose
  column dim is already `identity` has none.
- Stored groupings are user-supplied only. They live in
  `/{track}/_meta/column_groupings` (one variable `<group_dim>_membership`
  per group dim) and are written with `write_column_groupings`. They always
  contain an all-True `ALL` set (`ensure_all_set`) and cannot use the
  reserved dim `identity`.
- A summary track's column dim is the group dim of the grouping it was
  computed from (`identity` when there is no user grouping), and
  `perbase:source_grouping_hash` (`grouping_hash`) detects staleness.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-10-28"

import hashlib
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
import xarray as xr

from pbview import config

GROUPING_NAME = "column_groupings"
IDENTITY_DIM = "identity"


def grouping_hash(grouping: xr.DataArray) -> str:
    """Compute a hash for a (column, group) membership DataArray"""
    column_dim, group_dim = grouping.dims
    payload = b"|".join(
        [
            grouping.values.tobytes(),
            *(str(s).encode() for s in grouping[column_dim].values),
            *(str(s).encode() for s in grouping[group_dim].values),
        ]
    )
    return "md5:" + hashlib.md5(payload).hexdigest()


def read_sampleset_info(
    path,
    columns: Iterable[str],
    *,
    column_dim: str = "sample",
    group_dim: str = "sample_set",
) -> xr.DataArray:
    """Read a two-column (sample, sample_set) table as a membership with `ALL` first.

    Rows for labels not in `columns` are ignored; a column listed in no
    set belongs only to `ALL`. A column may be in several sets.
    """
    columns = [str(c) for c in columns]
    df = pd.read_table(
        path, header=None, sep=r"\s+", names=[column_dim, group_dim], dtype=str
    )
    df = df[df[column_dim].isin(columns)]
    sets = grouping_sets_from_dataframe(df, column_dim=column_dim, group_dim=group_dim)
    return ensure_all_set(
        membership_from_sets(columns, sets, column_dim=column_dim, group_dim=group_dim)
    )


def grouping_sets_from_dataframe(
    df: pd.DataFrame,
    *,
    column_dim: str,
    group_dim: str,
) -> Mapping[str, frozenset]:
    """Convert a general grouping DataFrame to a mapping of group names to members.

    Group members in `column_dim` and map them to names in `group_dim`. The
    resulting mapping is immutable and can be used to create a grouping DataArray.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    if len(set([column_dim, group_dim]).difference(set(df.columns.values))) != 0:
        raise ValueError(
            f"Expected DataFrame containing columns ({column_dim}, "
            f"{group_dim}), got {list(df.columns.values)}"
        )
    sets = {k: frozenset(v[column_dim]) for k, v in df.groupby(group_dim)}
    return MappingProxyType(sets)


def membership_from_sets(
    columns: Iterable[str],
    sets: Mapping[str, Iterable[str]],
    *,
    column_dim: str,
    group_dim: str,
) -> xr.DataArray:
    """Determine membership of columns from sets."""
    columns = tuple(columns)
    members = {name: frozenset(m) for name, m in sets.items()}
    unknown = set().union(*members.values()) - set(columns)
    if unknown:
        raise ValueError(f"Unknown {column_dim} labels in groups: {unknown}")
    data = np.array(
        [[col in members[group] for group in members] for col in columns],
        dtype=bool,
    ).reshape(len(columns), len(members))
    return xr.DataArray(
        data,
        dims=(column_dim, group_dim),
        coords={column_dim: list(columns), group_dim: list(members)},
        name=GROUPING_NAME,
    )


def identity_grouping(
    columns: Iterable[str],
    column_dim: str,
    name: str = config.DEFAULT_GROUPING_SET,
) -> xr.DataArray:
    """Grouping with one set containing every column (derived)"""
    columns = [str(c) for c in columns]
    return membership_from_sets(
        columns, {name: columns}, column_dim=column_dim, group_dim=IDENTITY_DIM
    )


def ensure_all_set(
    membership: xr.DataArray,
    name: str = config.DEFAULT_GROUPING_SET,
) -> xr.DataArray:
    """Return membership with an all-True `name` set, added first. Idempotent."""
    column_dim, group_dim = membership.dims
    if name in membership[group_dim].values.tolist():
        if not bool(membership.sel({group_dim: name}).all()):
            raise ValueError(f"set {name!r} is reserved and must contain every column")
        return membership
    columns = membership[column_dim].values
    all_ = xr.DataArray(
        np.ones((len(columns), 1), dtype=bool),
        dims=membership.dims,
        coords={column_dim: columns, group_dim: [name]},
        name=membership.name,
    )
    return xr.concat([all_, membership], dim=group_dim)


def write_column_groupings(
    store, track: str, groupings: Iterable[xr.DataArray]
) -> None:
    """Write membership to /{track}/_meta/column_groupings"""
    groupings = tuple(groupings)
    for g in groupings:
        if g.ndim != 2:
            raise ValueError(f"Grouping must be 2-d, got {g.dims}")
        if g.dims[1] == IDENTITY_DIM:
            raise ValueError(f"group_dim {IDENTITY_DIM!r} is reserved; it is derived")
    if not groupings:
        return
    groupings = tuple(ensure_all_set(g) for g in groupings)
    ds = xr.merge(
        [g.rename(f"{g.dims[1]}_membership") for g in groupings],
        join="exact",
    )
    ds.to_zarr(store, group=f"{track}/_meta/{GROUPING_NAME}", mode="w")


def validate_grouping(
    grouping: xr.DataArray | None,
    track_column_dim: str,
    track_column_labels: np.ndarray,
    track_dims: Iterable[str],
) -> xr.DataArray | None:
    if grouping is None:
        return None
    if grouping.ndim != 2 or grouping.dims[0] != track_column_dim:
        raise ValueError(
            f"grouping dims must be ({track_column_dim!r}, <group_dim>), "
            f"got {grouping.dims}"
        )
    group_dim = grouping.dims[1]
    if group_dim in set(track_dims):
        raise ValueError(f"group dim {grouping.dims[1]!r} collides with a track dim")
    if group_dim == IDENTITY_DIM:
        raise ValueError(f"group dim {IDENTITY_DIM!r} is reserved; it is derived")
    all_name = config.DEFAULT_GROUPING_SET
    if all_name not in grouping[group_dim].values.tolist() or not bool(
        grouping.sel({group_dim: all_name}).all()
    ):
        raise ValueError(
            f"grouping {group_dim!r} must contain a set {all_name!r} "
            "with all True values"
        )
    if grouping.dtype != bool:
        raise ValueError(f"grouping must be boolean, got {grouping.dtype}")
    if not np.array_equal(grouping[track_column_dim].values, track_column_labels):
        raise ValueError(f"grouping {track_column_dim!r} labels do not match the track")
    return grouping


def validate_track_groupings(
    groupings: Iterable[xr.DataArray] | None,
    track_column_dim: str,
    track_column_labels: np.ndarray,
    track_dims: Iterable[str],
) -> Mapping[str, xr.DataArray]:
    """Validate a list of track groupings.

    A grouping is an xr.DataArray that maps the columns of a track (
    source) to a set of groups (target). The source dimension must
    have the same column dimension as the track, and the target
    dimension must not collide with any of the track dimensions.

    Args:
        groupings: Iterable of xarray DataArrays representing groupings.
        track_column_dim: The name of the column dimension in the track.
        track_dims: Iterable of dimension names (row, column) in the track.

    Returns:
        A dictionary mapping grouping names to validated xarray DataArrays.
    """
    track_dims = tuple(track_dims)
    out = dict[str, xr.DataArray]()
    if groupings is None:
        return MappingProxyType(out)
    for g in groupings:
        g = validate_grouping(g, track_column_dim, track_column_labels, track_dims)
        if g is None:
            continue
        group_dim = g.dims[1]
        if group_dim in out:
            raise ValueError(f"Duplicate group dim {group_dim!r} in groupings")
        out[group_dim] = g
    return MappingProxyType(out)
