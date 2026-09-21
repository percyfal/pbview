import numpy as np
import pytest

from pbview.model.datastore import DataStore
from pbview.preprocess.track import compute_track_missingness, compute_track_sum


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def track(ds):
    return ds.tracks["depth"]


@pytest.fixture()
def coord(ds):
    return ds.base_coord


def test_compute_track_sum(track, coord):
    ds = compute_track_sum(track, coord)
    assert np.array_equal(ds["values"].values[-1, :], np.array([107, 50, 32, 25]))
    values = ds["values"].values[-2, :]
    assert np.sum(values[1:]) == values[0]


def test_compute_track_missingness(track, coord):
    ds = compute_track_missingness(track, coord)
    assert np.array_equal(ds["values"].values[0, :], np.array([7, 3, 2, 2]))
    assert np.array_equal(ds["values"].values[-1, :], np.array([0, 0, 0, 0]))
    ds = compute_track_missingness(track, coord, missing_cutoff=12)
    assert np.array_equal(ds["values"].values[0, :], np.array([7, 3, 2, 2]))
    assert np.array_equal(ds["values"].values[-1, :], np.array([2, 1, 0, 1]))
