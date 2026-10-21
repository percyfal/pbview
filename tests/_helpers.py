import hashlib

import numpy as np
import xarray as xr


def genome_checksum(contigs, offsets):
    """Compute a checksum for a list of contigs and offsets"""
    records = sorted(zip(contigs, np.diff(offsets)), key=lambda item: item[0].encode())
    payload = "".join(f"{name}\t{int(length)}\n" for name, length in records)
    return "md5:" + hashlib.md5(payload.encode()).hexdigest()


def grouping_hash(grouping: xr.DataArray) -> str:
    """Compute a hash for a grouping DataArray"""
    payload = (
        grouping.values.tobytes()
        + b"|"
        + b"|".join(s.encode() for s in grouping.coords["sample"].values)
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


def attach_sum(tree: xr.DataTree, track_name: str = "depth") -> xr.DataTree:
    """Attach sum track to datatree"""
    values = tree[track_name]["values"]
    dt_coords = tree[track_name].coords
    sample_sets = tree[f"{track_name}/_meta/column_grouping"]["values"]

    data = np.concat(
        [
            values[:, sample_sets.sel(sample_set=ss).values].sum(axis=1)
            for ss in sample_sets.sample_set.values
        ]
    )

    ds = xr.Dataset(
        data_vars={
            "values": (
                ("position", "sample_set"),
                data.reshape(-1, len(sample_sets.sample_set.values)),
            )
        },
        coords={
            "sample_set": ("sample_set", sample_sets.sample_set.values),
            "contigs": ("contig", dt_coords["contigs"].values),
            "offsets": ("contig_boundary", dt_coords["offsets"].values),
        },
    )
    track_attrs = {
        "perbase:provenance": "attach_sum",
        "perbase:column_dim": "sample_set",
        "perbase:source_track": track_name,
        "perbase:source_grouping_hash": tree[track_name].attrs["perbase:grouping_hash"],
        "perbase:genome_checksum": tree[track_name].attrs["perbase:genome_checksum"],
    }

    ds = update_dataset_attrs(ds, **track_attrs)
    tree[f"{track_name}_sum"] = ds
    return tree


def attach_missingness(
    tree: xr.DataTree, missing_cutoff: int = 3, track_name: str = "depth"
) -> xr.DataTree:
    """Attach missingness track to datatree"""
    values = tree[track_name]["values"]
    dt_coords = tree[track_name].coords
    sample_sets = tree[f"{track_name}/_meta/column_grouping"]["values"]

    data = np.concat(
        [
            (values[:, sample_sets.sel(sample_set=ss).values] <= missing_cutoff).sum(
                axis=1
            )
            for ss in sample_sets.sample_set.values
        ]
    )

    ds = xr.Dataset(
        data_vars={
            "values": (
                ("position", "sample_set"),
                data.reshape(-1, len(sample_sets.sample_set.values)),
            )
        },
        coords={
            "sample_set": ("sample_set", sample_sets.sample_set.values),
            "contigs": ("contig", dt_coords["contigs"].values),
            "offsets": ("contig_boundary", dt_coords["offsets"].values),
        },
    )
    track_attrs = {
        "perbase:provenance": "attach_missingness",
        "perbase:column_dim": "sample_set",
        "perbase:source_track": "depth",
        "perbase:source_grouping_hash": tree["depth"].attrs["perbase:grouping_hash"],
        "perbase:genome_checksum": tree["depth"].attrs["perbase:genome_checksum"],
    }

    ds = update_dataset_attrs(ds, **track_attrs)
    tree[f"depth_missing_cutoff={missing_cutoff}"] = ds
    return tree


def attach_default_sample_set(
    tree: xr.DataTree, track_name: str = "depth"
) -> xr.DataTree:
    """Attach default sample set to datatree metadata column_grouping slot"""
    samples = tree[track_name].coords["sample"].values
    values = xr.DataArray(
        np.array([[True]] * len(samples)),
        dims=("sample", "sample_set"),
        coords={
            "sample": tree[track_name].coords["sample"],
            "sample_set": ["ALL"],
        },
    )
    ss = xr.Dataset({"values": values})
    tree[f"{track_name}/_meta/column_grouping"] = ss
    tree[track_name].attrs["perbase:grouping_hash"] = grouping_hash(values)
    return tree


def tiny_datatree(track_name: str = "depth") -> xr.DataTree:
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

    contigs = np.array(["chr1", "chr2", "chr3"])
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
