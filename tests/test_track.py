import _helpers
import numpy as np
import pytest
import xarray as xr

from pbview.model.coordinates import ColumnSelection, Coordinates, PositionIntervals
from pbview.model.track import Track, _isel_intervals, resolve_columns

N = 24


@pytest.fixture
def dataset() -> xr.Dataset:
    """Dataset fixture"""
    values = np.arange(2 * N).reshape(2, N)
    return xr.Dataset(
        {"values": (("sample", "position"), values)},
        coords={
            "sample": np.array(["s1", "s2"]),
            "contigs": np.array(["chr1", "chr2", "chr10"]),
            "offsets": np.array([0, 10, 18, 24]),
        },
    )


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
def track(full_ds):
    return full_ds.tracks["depth"]


@pytest.fixture()
def coord(full_ds):
    return Coordinates(full_ds)


@pytest.fixture()
def genome(full_ds):
    return full_ds.genome


# FIXME: coord.samples=[] returns empty sample selection, not universe
def test_default_coord(track, coord):
    assert track.name == "depth"
    assert track.data(coord).sizes["position"] == 24
    assert track.data(coord).sizes["sample"] == 0  # FIXME: 7
    assert track.data(coord).sizes["contig"] == 3
    assert track.data(coord).sizes["contig_boundary"] == 4


def test_data_with_contigs(track, coord):
    assert track.data(coord=coord.with_contigs(["chr1"])).sizes["position"] == 10
    assert track.data(coord=coord.with_contigs(["chr2"])).sizes["position"] == 8
    with pytest.raises(KeyError):
        track.data(coord=coord.with_contigs(["chr4"]))


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


@pytest.mark.parametrize(
    "contigs,expected",
    [
        (None, 4),
        (("chr1",), 2),
    ],
)
def test_coverage_hist(contigs, expected, track, coord):
    bins = np.arange(0, 11)
    data, _ = track.coverage_hist(bins=bins, coord=coord.with_contigs(contigs))
    assert data[0] == expected


@pytest.mark.parametrize("cutoff,expected", [(0, 3), (2, 3), (10, 3)])
@pytest.mark.xfail(strict=False, reason="# REFACTOR: function returns one value?!?")
def test_missingness_hist(cutoff, expected, track, coord):
    data, _ = track.missingness_hist(coord=coord, missing_cutoff=cutoff)
    assert data[-1] == expected


def test_optimal_chunking(track):
    cs = track._optimal_chunks("sample")
    assert cs["sample"] == -1
    cs = track._optimal_chunks("position")
    assert cs["position"] == -1


def test_contig_mean_coverage(track):
    np.testing.assert_allclose(
        track._pre.contig_mean_coverage("ALL"),
        [86.5, 42.625, 31.333333],
        rtol=1e-5,
    )


def test_from_tree_finds_meta(sample_sets):
    raw = xr.DataTree.from_dict(
        {
            "/depth": xr.Dataset(),
            "/depth/_meta/column_grouping": sample_sets,
        }
    )
    track = Track.from_tree("depth", raw["depth"], raw)
    assert "column_grouping" in track.meta.data_vars


# FIXME: parametrize these tests on Dataset / DataArray
def test_whole_genome_is_noop(dataset):
    result = _isel_intervals(dataset, PositionIntervals(((0, N),)))
    xr.testing.assert_equal(result, dataset)
    result = _isel_intervals(dataset["values"], PositionIntervals(((0, N),)))
    xr.testing.assert_equal(result, dataset["values"])


def test_single_interval(dataset):
    result = _isel_intervals(dataset, PositionIntervals(((10, 18),)))
    assert result.sizes["position"] == 8
    np.testing.assert_array_equal(result["values"], dataset["values"].values[:, 10:18])
    result = _isel_intervals(dataset["values"], PositionIntervals(((10, 18),)))
    assert result.sizes["position"] == 8
    np.testing.assert_array_equal(result, dataset["values"].values[:, 10:18])


def test_non_adjacent_intervals_concat_in_order(dataset):
    result = _isel_intervals(dataset, PositionIntervals(((0, 10), (18, 24))))
    expected = np.concatenate(
        [dataset["values"].values[:, 0:10], dataset["values"].values[:, 18:24]], axis=1
    )
    assert result.sizes["position"] == 16
    np.testing.assert_array_equal(result["values"], expected)
    assert list(result["sample"].values) == ["s1", "s2"]


def test_empty_intervals(dataset):
    result = _isel_intervals(dataset, PositionIntervals(()))
    assert result.sizes["position"] == 0
    assert "values" in result.data_vars


@pytest.mark.parametrize("sels", [(()), (tuple([ColumnSelection("sample")]))])
def test_resolve_columns_empty_selection(sels):
    universe = (
        "s1",
        "s2",
        "s3",
    )
    result = resolve_columns(
        selections=sels,
        column_dim="sample",
        universe=universe,
    )
    assert result == universe


@pytest.mark.parametrize(
    "sels,expected",
    [
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        labels=(
                            "s1",
                            "s3",
                        ),
                    )
                ]
            ),
            ("s1", "s3"),
        ),
        (tuple([ColumnSelection("sample", labels=())]), ()),
    ],
)
def test_resolve_columns_sample_selection(sels, expected):
    universe = ("s1", "s2", "s3")
    result = resolve_columns(selections=sels, column_dim="sample", universe=universe)
    assert result == expected


def test_resolve_columns_unknown_raises():
    universe = ("s1", "s2", "s3")
    with pytest.raises(ValueError):
        resolve_columns(
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        labels=(
                            "s1",
                            "s4",
                        ),
                    )
                ]
            ),
            "sample",
            universe,
        )


def test_resolve_columns_projection_without_grouping_raises():
    with pytest.raises(ValueError):
        resolve_columns(
            tuple([ColumnSelection("sample_set", labels=("pop1",))]),
            "sample",
            ("s1", "s2", "s3"),
        )


@pytest.fixture
def tiny_grouping():
    samples = ["s1", "s2", "s3", "s4", "s5"]
    sample_sets = ["pop1", "pop2"]
    data = np.array([[True, False]] * 3 + [[False, True]] * 2)
    grouping = xr.DataArray(
        data=data,
        dims=("sample", "sample_set"),
        coords={"sample": samples, "sample_set": sample_sets},
    )
    return samples, sample_sets, grouping


@pytest.mark.parametrize(
    "sels,expected",
    [
        (tuple([ColumnSelection("sample_set", labels=("pop1",))]), ("s1", "s2", "s3")),
        (tuple([ColumnSelection("sample_set", labels=("pop2",))]), ("s4", "s5")),
    ],
)
def test_resolve_columns_sample_set_to_sample_projection(sels, expected, tiny_grouping):
    samples, _, grouping = tiny_grouping
    result = resolve_columns(sels, "sample", samples, grouping)
    assert result == expected


def test_resolve_columns_sample_to_sample_set_projection_is_noop(tiny_grouping):
    _, sample_sets, grouping = tiny_grouping
    result = resolve_columns(
        tuple([ColumnSelection("sample", ("s1", "s2", "s3"))]),
        "sample_set",
        sample_sets,
        grouping,
    )
    assert list(result) == sample_sets


@pytest.mark.parametrize(
    "sels,expected",
    [
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        (
                            "s1",
                            "s2",
                        ),
                    ),
                    ColumnSelection("sample_set", ("pop1",)),
                ]
            ),
            ("s1", "s2"),
        ),
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        (
                            "s1",
                            "s2",
                        ),
                    ),
                    ColumnSelection("sample_set", ("pop2",)),
                ]
            ),
            (),
        ),
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        (
                            "s1",
                            "s2",
                            "s4",
                        ),
                    ),
                    ColumnSelection("sample_set", ("pop2",)),
                ]
            ),
            ("s4",),
        ),
    ],
)
def test_resolve_columns_sample_and_sample_set_selection(sels, expected, tiny_grouping):
    samples, _, grouping = tiny_grouping
    result = resolve_columns(sels, "sample", samples, grouping)
    assert result == expected


@pytest.mark.parametrize(
    "sels,expected",
    [
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        (
                            "s1",
                            "s2",
                        ),
                    ),
                    ColumnSelection("sample_set", ("pop1",)),
                ]
            ),
            ("s1", "s2"),
        ),
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        (
                            "s1",
                            "s2",
                        ),
                    ),
                    ColumnSelection("sample_set", ("pop2",)),
                ]
            ),
            (),
        ),
        (
            tuple(
                [
                    ColumnSelection(
                        "sample",
                        (
                            "s1",
                            "s2",
                            "s4",
                        ),
                    ),
                    ColumnSelection("sample_set", ("pop2",)),
                ]
            ),
            ("s4",),
        ),
    ],
)
def test_track_resolve_columns_delegation(track, sels, expected):
    result = track.resolve_columns(sels)
    assert result == expected
