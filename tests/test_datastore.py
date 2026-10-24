from pathlib import Path

import dask.array as da
import numpy as np
import pbzarr
import pytest
import zarr

from pbview.model.datastore import DataStore
from pbview.model.track import Track

INIT_SCENARIOS = {
    "raw": {},
    "with_default_sample_set": {"with_default_sample_set": True},
    "with_sample_sets": {"sample_sets": True},
    "with_sum": {"with_sum": True},
    "with_sum_and_sample_sets": {"with_sum": True, "sample_sets": True},
    "with_missing": {"with_missing": True},
    "with_missing_and_sample_sets": {"with_missing": True, "sample_sets": True},
    "with_sum_and_missing": {"with_sum": True, "with_missing": True},
    "with_all": {"with_sum": True, "with_missing": True, "sample_sets": True},
}


@pytest.mark.parametrize("scenario", INIT_SCENARIOS)
def test_datastore_init(make_datastore_path, scenario, sample_sets):
    kwargs = INIT_SCENARIOS[scenario]
    if kwargs.get("sample_sets"):
        kwargs["sample_sets"] = sample_sets
    path = make_datastore_path(**kwargs)
    ds = DataStore(path)
    assert isinstance(ds, DataStore)
    assert isinstance(ds.tracks, dict)
    assert len(ds.tracks) > 0
    for track_name, track in ds.tracks.items():
        assert isinstance(track_name, str)
        assert isinstance(track, Track)
    assert len(ds.genome) == 24
    assert len(ds.genome.contigs) == 3


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


def test_datastore_genome_coords(raw_ds):
    assert list(raw_ds.contigs) == [
        "chr1",
        "chr2",
        "chr3",
    ]
    assert list(raw_ds.offsets) == [
        0,
        10,
        18,
        24,
    ]
