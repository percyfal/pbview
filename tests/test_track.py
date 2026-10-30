import _helpers
import numpy as np
import pytest
import xarray as xr

from pbview import config
from pbview.model.coordinates import (
    ColumnSelection,
    Coordinates,
    PositionIntervals,
    ResolvedSelection,
)
from pbview.model.grouping import IDENTITY_DIM, ensure_all_set
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


def test_default_coord(track, coord, genome):
    assert track.name == "depth"
    data = track.select(coord.with_samples(None).resolve_for(track, genome))
    assert data.sizes["position"] == 24
    assert data.sizes["sample"] == 7
    assert data.sizes["contig"] == 3
    assert data.sizes["contig_boundary"] == 4


def test_select_with_contigs(track, coord, genome):
    def n_positions(c):
        return track.select(c.resolve_for(track, genome)).sizes["position"]

    assert n_positions(coord.with_contigs(["chr1"])) == 10
    assert n_positions(coord.with_contigs(["chr2"])) == 8
    with pytest.raises(KeyError):
        coord.with_contigs(["chr4"]).resolve_for(track, genome)


def test_select_columns_and_positions(track, coord, genome):
    c = coord.with_sample_sets(["pop2"]).with_contigs(["chr2"])
    data = track.select(c.resolve_for(track, genome))
    assert data["sample"].values.tolist() == ["s4", "s5"]
    assert data["values"].shape == (8, 2)  # (position, sample)


def test_select_empty_columns(track, coord, genome):
    data = track.select(coord.with_samples(()).resolve_for(track, genome))
    assert data.sizes["sample"] == 0


def test_select_rejects_other_column_dim(track):
    resolved = ResolvedSelection(PositionIntervals(((0, 24),)), "sample_set", ("ALL",))
    with pytest.raises(ValueError, match="column dim"):
        track.select(resolved)


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
        (None, 3),
        (("chr1",), 1),
    ],
)
def test_coverage_hist(contigs, expected, track, coord):
    bins = np.arange(0, 11)
    data, _ = track.coverage_hist(bins=bins, coord=coord.with_contigs(contigs))
    assert data[0] == expected


@pytest.mark.parametrize("cutoff,expected", [(0, 3), (2, 3), (10, 3)])
def test_missingness_hist(cutoff, expected, track, coord):
    data, _ = track.missingness_hist(coord=coord, missing_cutoff=cutoff)
    assert data[-1] == expected


def test_optimal_chunking(track):
    cs = track._optimal_chunks("sample")
    assert cs["sample"] == -1
    cs = track._optimal_chunks("position")
    assert cs["position"] == -1


def test_contig_mean_sum_coverage(track):
    np.testing.assert_allclose(
        track._pre.contig_mean_coverage("ALL"),
        [135.6, 118.25, 106.16666667],
        rtol=1e-5,
    )


def test_from_tree_finds_meta(sample_sets):
    ss = xr.Dataset({"sample_set_membership": ensure_all_set(sample_sets)})
    depth = xr.Dataset(
        {"depth": (("position", "sample"), np.zeros((24, 7), dtype=np.uint32))},
        coords={"sample": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]},
    )

    raw = xr.DataTree.from_dict(
        {
            "/depth": depth,
            "/depth/_meta/column_groupings": ss,
        }
    )
    track = Track.from_tree(name="depth", data=raw["depth"], raw=raw)
    assert list(track.groupings.keys()) == ["identity", "sample_set"]
    da = track.groupings["sample_set"]
    assert da.name == "sample_set_membership"
    assert da.dims == ("sample", "sample_set")


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
    return samples, sample_sets, {"sample_set": grouping}


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


def test_all_set_is_identity(track):
    sel = ColumnSelection("sample_set", ("ALL",))
    assert track.resolve_columns((sel,)) == tuple(track.columns)


@pytest.mark.parametrize("name", ["depth", "depth_sum", "depth_missing_cutoff=3"])
def test_identity_selects_everything(full_ds, name):
    track = full_ds.tracks[name]
    sel = ColumnSelection(IDENTITY_DIM, (config.DEFAULT_GROUPING_SET,))
    assert track.resolve_columns((sel,)) == tuple(track.columns)


@pytest.mark.parametrize(
    "sets,expected",
    [(["pop1"], 3), (["pop1", "pop3"], 5), (["ALL"], 7), (None, 7)],
)
def test_n_selected_by_sample_sets(track, coord, sets, expected):
    assert track.n_selected(coord.with_sample_sets(sets).column) == expected


def test_n_selected_unknown_set_raises(track, coord):
    with pytest.raises(ValueError, match="unknown sample_set labels"):
        track.n_selected(coord.with_sample_sets(["pop0"]).column)


def test_n_selected_intersects_dims(track, coord):
    both = coord.with_samples(["s1", "s2"])
    assert track.n_selected(both.with_sample_sets(["pop2"]).column) == 0
    assert track.n_selected(both.with_sample_sets(["pop1"]).column) == 2


def test_groups(track, coord):
    assert track.groups() == ("ALL", "pop1", "pop2", "pop3")
    pop2 = coord.with_sample_sets(["pop2"]).column
    assert track.groups("sample_set", pop2) == ("ALL", "pop2")


def test_groups_without_sets_fall_back_to_identity(raw_ds):
    assert raw_ds.tracks["depth"].groups() == ("ALL",)


def test_matching_group(track, coord):
    assert track.matching_group(()) == "ALL"
    assert track.matching_group(coord.with_sample_sets(["pop2"]).column) == "pop2"
    assert track.matching_group(coord.with_samples(["s4", "s5"]).column) == "pop2"
    assert track.matching_group(coord.with_samples(["s1"]).column) is None


def test_overlapping_sets_on_track(make_datastore, sample_sets):
    ss = sample_sets.copy(deep=True)
    ss[0, 1] = True  # s1 in pop1 and pop2
    track = make_datastore(sample_sets=ss).tracks["depth"]
    sel = (ColumnSelection("sample_set", ("pop2",)),)
    assert track.resolve_columns(sel) == ("s1", "s4", "s5")
    assert track.groups("sample_set", sel) == ("ALL", "pop1", "pop2")


def test_n_selected_unknown_samples_raise(track, coord):
    with pytest.raises(ValueError, match="unknown sample labels"):
        track.n_selected(coord.with_samples(["s1", "s8"]).column)


def test_n_selected_empty_selection(track, coord):
    assert track.n_selected(coord.with_samples(()).column) == 0


def test_missing_grouping_raises_for_projection(raw_ds):
    track = raw_ds.tracks["depth"]
    coord = Coordinates(raw_ds)
    assert track.n_selected(coord.with_sample_sets(["ALL"]).column) == 7
    with pytest.raises(ValueError, match="no 'sample_set' grouping"):
        track.n_selected(coord.with_sample_sets(["pop1"]).column)


def test_selection_not_applicable_to_track_is_skipped(full_ds):
    summary = full_ds.tracks["depth_sum"]  # column dim sample_set
    sel = (ColumnSelection("sample", ("s1",)),)
    assert summary.resolve_columns(sel) == tuple(summary.columns)


def test_resolve_columns_projection_pair_needs_grouping():
    sel = (ColumnSelection("sample_set", ("pop1",)),)
    with pytest.raises(ValueError, match="no 'sample_set' grouping"):
        resolve_columns(sel, "sample", ["s1", "s2"], {})
    other = (ColumnSelection("tissue", ("x",)),)  # not a projection pair
    assert resolve_columns(other, "sample", ["s1"], {}) == ("s1",)
