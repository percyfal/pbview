from pathlib import Path

import dask.array as da
import numpy as np
import pbzarr
import pytest
import zarr

from pbview.datastore import DataStore, Track


@pytest.fixture
def zarr_group(tmpdir_factory):
    p = Path(tmpdir_factory.mktemp("zarr"))
    zarr_group = zarr.group(store=str(p))
    zarr_group.create_array("float_data", data=np.random.random(100).astype(np.float32))
    zarr_group.create_array("int_data", data=np.random.randint(0, 100, 100))
    return p


@pytest.fixture
def datatree():
    depth = (10 * da.random.random((100, 10), chunks=(5, 5))).astype(np.uint8)
    return depth


def test_empty_store(tmpdir_factory):
    p = Path(tmpdir_factory.mktemp("store"))
    with pytest.raises(zarr.errors.GroupNotFoundError):
        DataStore(path=p)


def test_generic_zarr_store(zarr_group):
    with pytest.raises(pbzarr.PbzError):
        DataStore(path=zarr_group)


# FIXME: parameterize on sampleinfo
def test_datastore(store):
    ds = DataStore(path=store)
    assert isinstance(ds, DataStore)
    assert ds.path == Path(store)
    assert isinstance(ds.tracks, dict)
    assert len(ds.tracks) > 0
    for track_name, track in ds.tracks.items():
        assert isinstance(track_name, str)
        assert isinstance(track, Track)
