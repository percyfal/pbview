import numpy as np
import pytest

from pbview.datastore import DataStore, Track


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def track(ds):
    return Track("track", data=ds.store["depth"])


@pytest.fixture()
def coord(ds):
    return ds.base_coord


def test_data_base_coord(track, coord):
    assert track.name == "track"
    assert track.data(coord).dims["position"] == 2_700_000
    assert track.data(coord).dims["sample"] == 7
    assert track.data(coord).dims["contig"] == 3
    assert track.data(coord).dims["contig_boundary"] == 4


def test_data_with_contigs(track, coord):
    assert track.data(coord=coord.with_contigs(["chr1"])).dims["position"] == 1_000_000
    assert track.data(coord=coord.with_contigs(["chr2"])).dims["position"] == 900_000
    assert track.data(coord=coord.with_contigs(["chr4"])).dims["position"] == 0


def test_summary(track, coord):
    summary = track.summary(coord)
    assert summary["genome_size"][0] == 2_700_000
    assert summary["selected_size"][0] == 2_700_000
    assert summary["n_samples"][0] == 7
    assert summary["n_contigs"][0] == 3
    assert summary["n_contigs_selected"][0] == 3


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {"contigs": ["chr1"]},
            {
                "genome_size": 2_700_000,
                "selected_size": 1_000_000,
                "n_samples": 7,
                "n_contigs": 3,
                "n_contigs_selected": 1,
            },
        ),
        (
            {"lower": 950_000},
            {
                "genome_size": 2_700_000,
                "selected_size": 1_000_000,
                "n_samples": 7,
                "n_contigs": 3,
                "n_contigs_selected": 1,
            },
        ),
        (
            {"upper": 950_000},
            {
                "genome_size": 2_700_000,
                "selected_size": 1_700_000,
                "n_samples": 7,
                "n_contigs": 3,
                "n_contigs_selected": 2,
            },
        ),
    ],
)
def test_summary_with_contigs(track, coord, args, expected):
    coord = coord.with_contigs(**args)
    summary = track.summary(coord)
    for key, value in expected.items():
        assert summary[key][0] == value


def test_lru_cache(track, coord):
    bins = np.arange(0, 11)
    track.hist(bins=bins, coord=coord)
    assert track._hist_cached.cache_info().hits == 0
    track.hist(bins=bins, coord=coord)
    assert track._hist_cached.cache_info().hits == 1
    bins = np.arange(0, 10)
    track.hist(bins=bins, coord=coord)
    assert track._hist_cached.cache_info().hits == 1
    track.hist(bins=bins, coord=coord)
    assert track._hist_cached.cache_info().hits == 2


def test_hist(track, coord):
    bins = np.arange(0, 11)
    data = track.hist(bins=bins, coord=coord)
    assert data[0] == 98944
    data = track.hist(bins=bins, coord=coord.with_contigs(["chr1"]))
    assert data[0] == 64346


def test_hist_thresholded(track, coord):
    bins = np.arange(0, 11)
    data = track.hist(bins=bins, coord=coord, threshold=0)
    assert data[0] == 99285
    data = track.hist(bins=bins, coord=coord, threshold=2)
    assert data[0] == 130909
    data = track.hist(bins=bins, coord=coord, threshold=10)
