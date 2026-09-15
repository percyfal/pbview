import numpy as np
import pytest

from pbview.datastore import DataStore, Track
from pbview.preprocess.track import compute_track_sum


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def track(ds):
    return Track("track", data=ds.store["depth"])


@pytest.fixture()
def coord(ds):
    return ds.base_coord


def test_compute_track_sum(track, coord):
    ds = compute_track_sum(track, coord)
    assert np.array_equal(ds["values"].values[-1, :], np.array([107, 50, 32, 25]))
