import numpy as np
import pytest

from pbview.model.datastore import DataStore
from pbview.model.histogram import compute_threshold_defaults


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def track(ds, store):
    return ds.tracks["depth"]


@pytest.fixture()
def coord(ds):
    return ds.base_coord


def test_data_base_coord(track, coord):
    assert track.name == "depth"
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
    assert summary["genome_size_all"][0] == 2_700_000
    assert summary["genome_size"][0] == 2_700_000


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {"contigs": ["chr1"]},
            {
                "genome_size_all": 2_700_000,
                "genome_size": 1_000_000,
            },
        ),
    ],
)
def test_summary_with_contigs(track, coord, args, expected):
    coord = coord.with_contigs(**args)
    summary = track.summary(coord)
    for key, value in expected.items():
        assert summary[key][0] == value


@pytest.mark.parametrize(
    argnames="args,expected",
    argvalues=[
        (
            {"lower": 950_000},
            {
                "genome_size_all": 2_700_000,
                "genome_size": 1_000_000,
            },
        ),
        (
            {"upper": 950_000},
            {
                "genome_size_all": 2_700_000,
                "genome_size": 1_700_000,
            },
        ),
    ],
)
def test_summary_with_length_filters(track, coord, args, expected):
    coord = coord.with_length_filter(**args)
    summary = track.summary(coord)
    for key, value in expected.items():
        assert summary[key][0] == value


def test_lru_cache(track, coord):
    bins = np.arange(0, 11)
    track.coverage_hist(bins=bins, coord=coord)
    assert track._coverage_hist_cached.cache_info().hits == 0
    track.coverage_hist(bins=bins, coord=coord)
    assert track._coverage_hist_cached.cache_info().hits == 1
    bins = np.arange(0, 10)
    track.coverage_hist(bins=bins, coord=coord)
    assert track._coverage_hist_cached.cache_info().hits == 1
    track.coverage_hist(bins=bins, coord=coord)
    assert track._coverage_hist_cached.cache_info().hits == 2


def test_coverage_hist(track, coord):
    bins = np.arange(0, 11)
    data, _ = track.coverage_hist(bins=bins, coord=coord)
    assert data[0] == 98944
    data, _ = track.coverage_hist(bins=bins, coord=coord.with_contigs(["chr1"]))
    assert data[0] == 64346
    data, bins = track.coverage_hist(bins=bins, coord=coord.with_contigs(["chr1"]))


def test_missingness_hist(track, coord):
    bins = np.arange(0, 11)
    data, _ = track.missingness_hist(coord=coord, threshold=0)
    assert data[-1] == 98944
    data, _ = track.missingness_hist(coord=coord, threshold=2)
    assert data[-1] == 146622
    data, _ = track.missingness_hist(coord=coord, threshold=10)
    data, bins = track.missingness_hist(coord=coord, threshold=10)


def test_optimal_chunking(track):
    cs = track._optimal_chunks("sample")
    assert cs["sample"] == -1
    cs = track._optimal_chunks("position")
    assert cs["position"] == -1
