import _helpers
import numpy as np
import pytest
import xarray as xr

from pbview.model.datastore import DataStore
from pbview.model.track import Track


@pytest.fixture
def d1() -> xr.DataTree:
    """Track data fixture"""
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    contigs = np.array(["chr1", "chr2"])
    offsets = np.array([0, 2, 3])
    samples = np.array(["s1", "s2", "s3"])
    return _helpers.make_track_group(
        values=values, contigs=contigs, offsets=offsets, column_values=samples
    )


def test_track_init(d1):
    t = Track("test_track", d1)
    assert t.name == "test_track"
    assert t.column_dim == "sample"
    assert np.array_equal(t.columns, ["s1", "s2", "s3"])
    assert np.array_equal(t._data.position.values, [0, 1, 2])


@pytest.fixture()
def ds(store, sampleset_info):
    return DataStore(path=store, sampleset_info=sampleset_info)


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
    with pytest.raises(ValueError):
        track.data(coord=coord.with_contigs(["chr4"]))


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
    data, _ = track.missingness_hist(coord=coord, missing_cutoff=0)
    assert data[-1] == 98944
    data, _ = track.missingness_hist(coord=coord, missing_cutoff=2)
    assert data[-1] == 146622
    data, _ = track.missingness_hist(coord=coord, missing_cutoff=10)
    data, bins = track.missingness_hist(coord=coord, missing_cutoff=10)


def test_optimal_chunking(track):
    cs = track._optimal_chunks("sample")
    assert cs["sample"] == -1
    cs = track._optimal_chunks("position")
    assert cs["position"] == -1


def test_contig_mean_coverage(track):
    np.testing.assert_allclose(
        track._pre.contig_mean_coverage("ALL"),
        [75.17373, 115.60176, 123.73403],
        rtol=1e-5,
    )
