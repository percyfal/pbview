import numpy as np
import pytest

from pbview.model.datastore import DataStore
from pbview.model.grouping import IDENTITY_DIM, grouping_hash
from pbview.preprocess import preprocess as preprocess_mod
from pbview.preprocess.track import compute_track_missingness, compute_track_sum

SAMPLESET_INFO = "s1 pop1\ns2 pop1\ns3 pop1\ns4 pop2\ns5 pop2\ns6 pop3\ns7 pop3\n"


@pytest.fixture()
def track(full_ds):
    return full_ds.tracks["depth"]


@pytest.fixture()
def membership(track):
    return track.groupings["sample_set"]


def test_compute_track_sum(track, membership):
    ds = compute_track_sum(track, membership)
    assert ds["values"].dims == ("position", "sample_set")
    assert ds["sample_set"].values.tolist() == ["ALL", "pop1", "pop2", "pop3"]
    assert np.array_equal(ds["values"].values[-1, :], np.array([129, 51, 48, 30]))
    values = ds["values"].values[-2, :]
    assert np.sum(values[1:]) == values[0]
    assert ds.attrs["perbase:column_dim"] == "sample_set"
    assert ds.attrs["perbase:source_track"] == "depth"
    assert ds.attrs["perbase:source_grouping_hash"] == grouping_hash(membership)


def test_compute_track_missingness(track, membership):
    ds = compute_track_missingness(track, membership)
    assert np.array_equal(ds["values"].values[0, :], np.array([7, 3, 2, 2]))
    assert np.array_equal(ds["values"].values[-1, :], np.array([0, 0, 0, 0]))
    ds = compute_track_missingness(track, membership, missing_cutoff=12)
    assert np.array_equal(ds["values"].values[0, :], np.array([7, 3, 2, 2]))
    assert np.array_equal(ds["values"].values[-1, :], np.array([2, 1, 0, 1]))


def test_compute_track_sum_identity(raw_ds):
    track = raw_ds.tracks["depth"]
    ds = compute_track_sum(track, track.groupings[IDENTITY_DIM])
    assert ds["values"].dims == ("position", IDENTITY_DIM)
    assert ds["values"].values[-1, 0] == 129
    assert ds.attrs["perbase:column_dim"] == IDENTITY_DIM


def test_compute_matches_fixture_summaries(full_ds, track, membership):
    """Independent implementations (fixture helpers, preprocess) agree."""
    np.testing.assert_array_equal(
        compute_track_sum(track, membership)["values"].values,
        full_ds.tracks["depth_sum"].values.values,
    )
    np.testing.assert_array_equal(
        compute_track_missingness(track, membership, 3)["values"].values,
        full_ds.tracks["depth_missing_cutoff=3"].values.values,
    )


def test_preprocess_writes_groupings_and_summaries(make_datastore_path, tmp_path):
    path = make_datastore_path()
    info = tmp_path / "sets.tsv"
    info.write_text(SAMPLESET_INFO)
    preprocess_mod.preprocess(path, progress=False, sampleset_info=info)

    ds = DataStore(path)
    grouping = ds.tracks["depth"].groupings["sample_set"]
    assert grouping["sample_set"].values.tolist() == ["ALL", "pop1", "pop2", "pop3"]
    for name in ("depth_sum", "depth_missing_cutoff=3"):
        summary = ds.tracks[name]
        assert summary.column_dim == "sample_set"
        assert summary.attrs["perbase:source_grouping_hash"] == grouping_hash(grouping)


def test_preprocess_without_sampleset_info_uses_identity(make_datastore_path):
    path = make_datastore_path()
    preprocess_mod.preprocess(path, progress=False)
    summary = DataStore(path).tracks["depth_sum"]
    assert summary.column_dim == IDENTITY_DIM
    assert summary.columns == ["ALL"]


def test_compute_thresholds_runs(make_datastore_path, sample_sets, capsys):
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    preprocess_mod.compute_thresholds(path, progress=False)
    out = capsys.readouterr().out
    for name in ("ALL", "pop1", "pop2", "pop3"):
        assert name in out
