import hashlib

import numpy as np
import xarray as xr


def genome_checksum(contigs, offsets):
    """Compute a checksum for a list of contigs and offsets"""
    records = sorted(zip(contigs, np.diff(offsets)), key=lambda item: item[0].encode())
    payload = "".join(f"{name}\t{int(length)}\n" for name, length in records)
    return "md5:" + hashlib.md5(payload.encode()).hexdigest()


def update_dataset_attrs(ds, **kwargs):
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


def update_datatree_root_attrs(dt, **kwargs):
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


def make_datatree(values, contigs, offsets, samples):
    ds = xr.Dataset(
        data_vars={"values": (("position", "sample"), values.T)},
        coords={
            "sample": ("sample", samples),
            "contigs": ("contig", contigs),
            "offsets": ("contig_boundary", offsets),
        },
    )
    ds = update_dataset_attrs(ds)
    dt = update_datatree_root_attrs(xr.DataTree(ds))
    return dt
