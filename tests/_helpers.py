import hashlib

import numpy as np
import xarray as xr

from pbview.model.grouping import membership_from_sets, write_column_groupings


def genome_checksum(contigs, offsets):
    """Compute a checksum for a list of contigs and offsets"""
    records = sorted(zip(contigs, np.diff(offsets)), key=lambda item: item[0].encode())
    payload = "".join(f"{name}\t{int(length)}\n" for name, length in records)
    return "md5:" + hashlib.md5(payload.encode()).hexdigest()


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


def update_dataset_attrs(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    """Update dataset attributes with default values.

    Custom values can be provided as keyword arguments.
    """
    attrs = dict(
        {
            "zarr_conventions": [{"uuid": "test-perbase", "name": "perbase"}],
            "perbase:version": "0.4",
            "perbase:coordinates": "0-based-half-open",
            "perbase:genome_checksum": genome_checksum(
                ds.contigs.values, ds.offsets.values
            ),
        },
        **kwargs,
    )
    ds.attrs.update(attrs)
    return ds


def update_datatree_root_attrs(dt: xr.DataTree, **kwargs) -> xr.DataTree:
    """Update datatree root attributes with default values.

    Custom values can be provided as keyword arguments.
    """
    attrs = dict(
        {
            "zarr_conventions": [{"uuid": "test-perbase", "name": "perbase"}],
            "perbase:kind": "collection",
        },
        **kwargs,
    )
    dt.attrs.update(attrs)
    return dt


def make_track_group(
    *,
    values: np.ndarray,
    contigs: np.ndarray,
    offsets: np.ndarray,
    column_dim: str = "sample",
    column_values: np.ndarray = None,
) -> xr.DataTree:
    """Make track data group"""
    ds = xr.Dataset(
        data_vars={"values": (("position", column_dim), values.T)},
        coords={
            column_dim: (column_dim, column_values),
            "contigs": ("contig", contigs),
            "offsets": ("contig_boundary", offsets),
        },
    )
    dt = xr.DataTree(update_dataset_attrs(ds))
    return dt


def make_pbzarr_datatree(track_name: str = "depth", **kwargs):
    """Make pbzarr data tree with one dataset track"""
    ds = make_track_group(**kwargs)
    dt = xr.DataTree.from_dict({track_name: ds})
    dt = update_datatree_root_attrs(dt)
    return dt


def all_set(samples, name: str = "ALL") -> xr.DataArray:
    """Membership of samples (the identity grouping)"""
    samples = [str(s) for s in samples]
    return membership_from_sets(
        samples, {name: samples}, column_dim="sample", group_dim="sample_set"
    )


def with_all_set(
    sample_sets: xr.DataArray | None, samples, name: str = "ALL"
) -> xr.DataArray:
    """Add `ALL` sample set to existing sample sets, first column."""
    base = all_set(samples, name=name)
    if sample_sets is None:
        return base
    return xr.concat([base, sample_sets], dim="sample_set", join="exact")


def _attach_summary(
    tree: xr.DataTree,
    *,
    name: str,
    provenance: str,
    membership: xr.DataArray,
    reduce,
    track_name: str,
) -> xr.DataTree:
    """Attach a per-sample-set summary track reduced over `sample`."""
    values = tree[track_name]["values"].values  # (position, sample)
    coords = tree[track_name].coords
    # Get the group names from the sample_set column
    groups = membership["sample_set"].values
    data = np.stack(
        [reduce(values[:, membership.sel(sample_set=g).values]) for g in groups],
        axis=1,
    )
    ds = xr.Dataset(
        data_vars={"values": (("position", "sample_set"), data)},
        coords={
            "sample_set": ("sample_set", groups),
            "contigs": ("contig", coords["contigs"].values),
            "offsets": ("contig_boundary", coords["offsets"].values),
        },
    )
    tree[name] = update_dataset_attrs(
        ds,
        **{
            "perbase:provenance": provenance,
            "perbase:column_dim": "sample_set",
            "perbase:source_track": track_name,
            "perbase:source_grouping_hash": grouping_hash(membership),
            "perbase:genome_checksum": tree[track_name].attrs[
                "perbase:genome_checksum"
            ],
        },
    )
    return tree


def attach_sum(
    tree: xr.DataTree, membership: xr.DataArray, track_name: str = "depth"
) -> xr.DataTree:
    return _attach_summary(
        tree,
        name=f"{track_name}_sum",
        provenance="attach_sum",
        membership=membership,
        reduce=lambda x: x.sum(axis=1),
        track_name=track_name,
    )


def attach_missingness(
    tree: xr.DataTree,
    membership: xr.DataArray,
    missing_cutoff: int = 3,
    track_name: str = "depth",
) -> xr.DataTree:
    return _attach_summary(
        tree,
        name=f"{track_name}_missing_cutoff={missing_cutoff}",
        provenance="attach_missingness",
        membership=membership,
        reduce=lambda x: (x <= missing_cutoff).sum(axis=1),
        track_name=track_name,
    )


def make_store(
    path,
    *,
    default_set: bool = False,
    sample_sets: xr.DataArray | None = None,
    with_sum: bool = False,
    with_missing: bool = False,
    track_name: str = "depth",
):
    """Write a tiny pbzarr store. Groupings are built in memory"""
    tree = tiny_datatree(track_name=track_name)
    membership = None
    if default_set or sample_sets is not None or with_sum or with_missing:
        samples = tree[track_name].coords["sample"].values
        membership = with_all_set(sample_sets, samples)
        tree[track_name].attrs["perbase:grouping_hash"] = grouping_hash(membership)
    if with_sum:
        attach_sum(tree, membership, track_name=track_name)
    if with_missing:
        attach_missingness(tree, membership, track_name=track_name)
    tree.to_zarr(path)
    if membership is not None:
        write_column_groupings(path, track_name, [membership])
    return path


def tiny_datatree(track_name: str = "depth", contigs=None) -> xr.DataTree:
    """Create a tiny datatree for testing."""
    values = np.array(
        [
            [0, 5, 5, 5, 5, 4, 4, 4, 4, 4]
            + [0, 7, 7, 7, 7, 7, 7, 7]
            + [0, 12, 11, 11, 11, 12],
            [0, 28, 28, 28, 28, 28, 28, 27, 27, 27]
            + [0, 17, 17, 17, 17, 17, 17, 17]
            + [0, 17, 17, 17, 17, 17],
            [0, 33, 33, 34, 33, 34, 33, 34, 34, 34]
            + [0, 13, 13, 13, 13, 13, 12, 12]
            + [0, 21, 21, 21, 22, 22],
            [0, 29, 30, 30, 30, 30, 30, 31, 31, 31]
            + [0, 25, 25, 25, 25, 25, 25, 25]
            + [0, 26, 26, 26, 26, 26],
            [0, 24, 24, 24, 24, 25, 25, 25, 25, 25]
            + [0, 24, 24, 24, 24, 24, 24, 24]
            + [0, 22, 22, 22, 22, 22],
            [0, 7, 7, 9, 8, 9, 10, 11, 11, 11]
            + [0, 16, 16, 16, 16, 16, 16, 16]
            + [0, 8, 8, 7, 7, 7],
            [0, 21, 21, 21, 21, 21, 21, 21, 21, 21]
            + [0, 33, 32, 33, 34, 34, 34, 34]
            + [0, 21, 21, 23, 23, 23],
        ]
    )
    contigs = (
        np.array(contigs)
        if contigs is not None
        else np.array(["chr1", "chr2", "chr10"])
    )
    offsets = np.array([0, 10, 18, 24])
    samples = np.array(["s1", "s2", "s3", "s4", "s5", "s6", "s7"])
    dt = make_pbzarr_datatree(
        values=values, contigs=contigs, offsets=offsets, column_values=samples
    )
    return dt


def sample_sets() -> xr.DataArray:
    """Create a sample sets DataArray for testing."""
    sample_sets = xr.DataArray(
        np.array(
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [False, True, False],
                [False, True, False],
                [False, False, True],
                [False, False, True],
            ]
        ),
        dims=("sample", "sample_set"),
        coords={
            "sample": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
            "sample_set": ["pop1", "pop2", "pop3"],
        },
    )
    return sample_sets
