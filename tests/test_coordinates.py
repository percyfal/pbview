import warnings
from unittest.mock import MagicMock

import _helpers
import numpy as np
import pytest

from pbview import config
from pbview._compat import ShimWarning
from pbview.model.coordinates import (
    ColumnSelection,
    Coordinates,
    Genome,
    PositionIntervals,
    SpatialSelection,
    _legacy_grouping,
)
from pbview.model.datastore import DataStore
from pbview.model.grouping import IDENTITY_DIM, identity_grouping
from pbview.model.track import Track


@pytest.fixture
def data():
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    contigs = np.array(["chr1", "chr2"])
    offsets = np.array([0, 2, 3])
    samples = np.array(["s1", "s2", "s3"])
    return _helpers.make_track_group(
        values=values, contigs=contigs, offsets=offsets, column_values=samples
    )


@pytest.fixture
def genome():
    return Genome(contigs=("chr1", "chr2", "chr10"), offsets=(0, 10, 18, 24))


@pytest.fixture
def coords(raw_ds):
    return Coordinates(raw_ds)


@pytest.mark.parametrize(
    "position_intervals,expected",
    [
        (((0, 100), (200, 300)), 200),
        ((()), 0),
    ],
)
def test_position_intervals(position_intervals, expected):
    pi = PositionIntervals(position_intervals)
    assert pi.n_positions == expected


def test_genome_init():
    genome = Genome(contigs=("chr10", "chr1"), offsets=(0, 10, 18))
    assert genome.contigs == ("chr10", "chr1")
    assert genome.offsets == (0, 10, 18)
    assert genome.size == 18


def test_genome_init_raises():
    with pytest.raises(ValueError):
        _ = Genome(contigs=("chr1", "chr2"), offsets=(0, 10))


def test_genome_contig_lengths():
    genome = Genome(contigs=("chr1", "chr2"), offsets=(0, 10, 18))
    np.testing.assert_equal([10, 8], genome.lengths)


def test_genome_contigs_in_length_range(genome):
    contigs = genome.contigs
    np.testing.assert_equal(genome.contigs_in_length_range(), contigs)
    np.testing.assert_equal(genome.contigs_in_length_range(lower=6), contigs)
    np.testing.assert_equal(genome.contigs_in_length_range(lower=9), (contigs[0],))
    np.testing.assert_equal(genome.contigs_in_length_range(lower=11), ())
    np.testing.assert_equal(genome.contigs_in_length_range(upper=5), ())
    np.testing.assert_equal(genome.contigs_in_length_range(upper=9), contigs[1:3])
    np.testing.assert_equal(genome.contigs_in_length_range(upper=12), contigs)
    np.testing.assert_equal(
        genome.contigs_in_length_range(lower=7, upper=9), (contigs[1],)
    )


def test_genome_contig_indices(genome, coords):
    indices = genome.contig_indices(coords.with_contigs(()).spatial.contigs)
    assert len(indices) == 0
    indices = genome.contig_indices(coords.with_contigs(("chr2",)).spatial.contigs)
    assert indices == [1]
    indices = genome.contig_indices(coords.with_contigs(None).spatial.contigs)
    np.testing.assert_equal(indices, [0, 1, 2])


def test_coordinates_with_contigs_empty_means_nothing(coords, genome):
    coords = coords.with_contigs(contigs=())
    assert coords.spatial == SpatialSelection(())
    assert coords.selected_contigs(genome) == ()
    assert genome.resolve(coords.spatial).n_positions == 0


def test_coordinates_with_contigs_none_means_all(coords, genome):
    assert coords.with_contigs(None).selected_contigs(genome) == genome.contigs


@pytest.mark.parametrize(
    "contigs",
    [
        (
            "chr1",
            "chr2",
        )
    ],
)
def test_spatial_selection_init(contigs):
    sel = SpatialSelection(contigs=contigs)
    assert sel.contigs == contigs


def test_spatial_selection_none_is_noop():
    sel = SpatialSelection(contigs=None)
    assert sel.contigs is None


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({"contigs": ("chr2",)}, ("chr2",)),
        (
            {"contigs": ("chr10", "chr2")},
            (
                "chr2",
                "chr10",
            ),
        ),
        (
            {"contigs": ()},
            (),
        ),
    ],
)
def test_genome_resolve_contigs_for_spatial_selection(genome, kw, expected):
    sel = SpatialSelection(**kw)
    contigs = genome.resolve_contigs(sel)
    assert contigs == expected


def test_genome_resolve_contigs_for_spatial_selection_raise_wrong_contig(genome):
    with pytest.raises(KeyError):
        sel = SpatialSelection(contigs=("chr4",))
        genome.resolve_contigs(sel)


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({"contigs": ("chr10",)}, (6, ((18, 24),))),
        (
            {"contigs": ("chr10", "chr1")},
            (
                16,
                (
                    (0, 10),
                    (18, 24),
                ),
            ),
        ),
    ],
)
def test_genome_resolve(genome, kw, expected):
    sel = SpatialSelection(**kw)
    pi = genome.resolve(sel)
    assert isinstance(pi, PositionIntervals)
    assert pi.n_positions == expected[0]
    assert pi.intervals == expected[1]


@pytest.mark.parametrize("selected", [None, [], ()])
def test_empty_column_selection_is_noop(data, selected):
    sel = ColumnSelection(column_dim="sample", labels=selected)
    ret = sel.apply(data)
    assert ret == data


@pytest.mark.parametrize("selected", [None, [], ()])
def test_empty_column_selection_on_subset_is_noop(data, selected):
    data = data.copy()
    data = data.sel(sample=["s1"])
    ret = ColumnSelection(column_dim="sample", labels=selected).apply(data)
    assert ret == data


def test_column_selection_selected(data):
    sel = ColumnSelection(column_dim="sample", labels=["s1", "s3"])
    ret = sel.apply(data)
    assert ret.coords["sample"].values.tolist() == ["s1", "s3"]


@pytest.mark.xfail(
    strict=True,
    reason="REFACTOR: Coordinates will not know about sample sets going forward",
)
@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, 0),
        ({"with_sum": True, "with_missing": True}, 0),  # FIXME: should be 1
        (
            {
                "with_sum": True,
                "with_missing": True,
                "sample_sets": _helpers.sample_sets(),
            },
            3,
        ),
    ],
)
def test_coordinates_init(make_datastore, kw, expected):
    ds = make_datastore(**kw)
    coords = Coordinates(ds)
    assert coords.n_sample_sets == expected


def test_coordinates_init_with_column_selection(make_datastore):
    ds = make_datastore(sample_sets=_helpers.sample_sets())
    with pytest.raises(TypeError):
        Coordinates(ds, (), ColumnSelection())
    with pytest.raises(ValueError):
        Coordinates(ds, (), [ColumnSelection("sample"), ColumnSelection("sample")])


def test_coordinates_with_contigs(coords, genome):
    assert len(coords.with_contigs(["chr1", "chr2"]).selected_contigs(genome)) == 2
    assert len(coords.with_contigs(["chr2"]).selected_contigs(genome)) == 1
    assert (
        len(coords.with_contigs(["chr2"]).with_contigs().selected_contigs(genome)) == 3
    )


def test_coordinates_with_samples(coords):
    pass


def test_coordinates_with_sample_sets(coords):
    pass


def test_default_grouping_is_derived_identity(raw_ds):
    groupings = raw_ds.tracks["depth"].groupings
    assert list(groupings) == [IDENTITY_DIM]
    g = groupings[IDENTITY_DIM]
    assert g[IDENTITY_DIM].values.tolist() == [config.DEFAULT_GROUPING_SET]
    assert bool(g.all())


def test_legacy_grouping_ignores_identity():
    from types import SimpleNamespace

    cols = ["s1", "s2"]
    tracks = {
        n: SimpleNamespace(
            column_dim="sample",
            groupings={IDENTITY_DIM: identity_grouping(cols, "sample")},
        )
        for n in ("a", "b")
    }
    assert _legacy_grouping(SimpleNamespace(tracks=tracks)) is None


def test_user_sets_follow_all(full_ds):
    g = full_ds.tracks["depth"].groupings["sample_set"]
    assert g["sample_set"].values.tolist() == ["ALL", "pop1", "pop2", "pop3"]
    assert bool(g.sel(sample_set="ALL").all())


# Compose
@pytest.mark.parametrize(
    "kw,expected",
    [
        (
            {},
            (
                "chr1",
                "chr2",
                "chr10",
            ),
        ),
        (
            {"lower": 7},
            (
                "chr1",
                "chr2",
            ),
        ),
        (
            {"upper": 9},
            (
                "chr2",
                "chr10",
            ),
        ),
        ({"lower": 7, "upper": 8}, ("chr2",)),
    ],
)
def test_coordinates_with_composed_length_filter(coords, genome, kw, expected):
    keep = genome.contigs_in_length_range(**kw)
    np.testing.assert_equal(coords.with_contigs(keep).spatial.contigs, expected)


def test_coordinates_with_contigs_and_composed_length_filter(coords, genome):
    keep = genome.contigs_in_length_range(lower=9)
    assert len(coords.with_contigs(keep).spatial.contigs) == 1
    keep = genome.contigs_in_length_range(lower=9)
    assert len(coords.with_contigs(keep).spatial.contigs) == 1


def test_spatial_rejects_str():
    with pytest.raises(TypeError):
        SpatialSelection("chr1")


@pytest.fixture(params=["has_sample_sets", "no_sample_sets"])
def legacy_coords(make_datastore_path, request):
    if request.param == "has_sample_sets":
        path = make_datastore_path(sample_sets=_helpers.sample_sets())
    elif request.param == "no_sample_sets":
        path = make_datastore_path()
    return Coordinates(DataStore(path))


def test_with_contigs_does_not_warn(legacy_coords):
    legacy_coords = legacy_coords
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShimWarning)
        legacy_coords.with_contigs(["chr1"])


def test_with_samples_does_not_warn(legacy_coords):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShimWarning)
        legacy_coords.with_samples(["s1"]).with_sample_sets(["pop1"])


def test_with_samples(legacy_coords):
    coord = legacy_coords
    assert coord.with_samples(["s1", "s2"]).n_samples == 2
    assert coord.with_samples().n_samples == 7
    assert coord.with_samples(()).n_samples == 0


def test_with_samples_rejects_unknown_on_resolve(legacy_coords):
    coord = legacy_coords.with_samples(["s1", "s8"])  # construction does not validate
    with pytest.raises(ValueError, match="unknown sample labels"):
        coord.n_samples


def test_reset_clears_contigs(coords, genome):
    assert (
        coords.with_contigs(("chr1",)).reset().selected_contigs(genome)
        == genome.contigs
    )


def test_reset_from_empty_selection(coords, genome):
    assert coords.with_contigs(()).reset().selected_contigs(genome) == genome.contigs


def test_reset_is_default(coords):
    assert coords.with_contigs(("chr1",)).reset().spatial == SpatialSelection()


def test_legacy_coords_cache_hit(legacy_coords):
    coord = legacy_coords
    assert hash(coord) == hash(coord)
    assert hash(coord.with_sample_sets(["pop1"])) == hash(
        coord.with_sample_sets(["pop1"])
    )
    assert hash(coord) != hash(coord.with_sample_sets(["pop1"]))


@pytest.mark.parametrize(
    "contigs, expected",
    [
        (None, ((0, 24),)),  # all contigs collapse to one slice
        (("chr1", "chr2"), ((0, 18),)),  # adjacent contigs merge
        (("chr10", "chr1"), ((0, 10), (18, 24))),  # gap preserved, genome order
        ((), ()),
    ],
)
def test_resolve_merges_adjacent(genome, contigs, expected):
    assert genome.resolve(SpatialSelection(contigs)).intervals == expected


def test_resolve_does_not_swallow_one_bp_gap():
    g = Genome(("a", "b", "c"), (0, 5, 6, 10))  # b has length 1
    assert g.resolve(SpatialSelection(("a", "c"))).intervals == ((0, 5), (6, 10))


def test_eq_hash_follow_contigs(coords):
    a = coords.with_contigs(("chr1",))
    assert a == coords.with_contigs(("chr1",))
    assert hash(a) == hash(coords.with_contigs(("chr1",)))
    assert a != coords.with_contigs(("chr2",))


def test_hash_does_not_validate(coords):
    hash(coords.with_contigs(("chrX",)))  # no raise; validation is at resolve time


def test_length_range_within_subset(genome):
    assert genome.contigs_in_length_range(10, contigs=("chr10", "chr1", "chr2")) == (
        "chr1",
    )


def test_length_range_unknown_contig_raises(genome):
    with pytest.raises(KeyError):
        genome.contigs_in_length_range(0, contigs=("chrX",))


@pytest.mark.parametrize(
    "contigs,expected",
    [
        (None, ((0, 24),)),
        ((), ()),
        (("chr1",), ((0, 10),)),
        (
            (
                "chr1",
                "chr10",
            ),
            (
                (0, 10),
                (18, 24),
            ),
        ),
    ],
)
def test_coordinates_resolve_for_contigs(contigs, expected, coords, genome):
    track = MagicMock(spec_set=Track)
    track.columns = ["s1", "s2", "s3"]
    track.column_dim = "sample"
    results = coords.with_contigs(contigs).resolve_for(track, genome)
    assert results.intervals.intervals == expected


def test_with_columns_none_clears(coords):
    c = coords.with_columns(ColumnSelection("sample", ("s1",)))
    assert c.with_columns(ColumnSelection("sample", None)) == coords.reset()


@pytest.fixture
def set_coords(full_ds):
    return Coordinates(full_ds)


@pytest.mark.parametrize(
    "sets,expected",
    [(["pop1"], 3), (["pop1", "pop3"], 5), (["ALL"], 7), (None, 7)],
)
def test_with_sample_sets_counts(set_coords, sets, expected):
    assert set_coords.with_sample_sets(sets).n_samples == expected


def test_with_sample_sets_unknown_rejected_on_resolve(set_coords):
    coord = set_coords.with_sample_sets(["pop0"])
    with pytest.raises(ValueError, match="unknown sample_set labels"):
        coord.n_samples


def test_with_sample_sets_without_sets(coords):
    assert coords.with_sample_sets(["ALL"]).n_samples == 7
    with pytest.raises(ValueError, match="no 'sample_set' grouping"):
        coords.with_sample_sets(["pop1"]).n_samples


def test_samples_and_sample_sets_intersect(set_coords):
    both = set_coords.with_samples(["s1", "s2"])
    assert both.with_sample_sets(["pop2"]).n_samples == 0
    assert both.with_sample_sets(["pop1"]).n_samples == 2


def test_with_samples_replaces_earlier_selection(set_coords):
    coord = set_coords.with_samples(["s1", "s2"]).with_samples(["s2", "s3"])
    assert coord.samples.tolist() == ["s2", "s3"]


def test_sample_selections_are_part_of_the_key(set_coords):
    a = set_coords.with_sample_sets(["pop1"])
    assert a == set_coords.with_sample_sets(["pop1"])
    # same samples, different selection
    assert a != set_coords.with_samples(["s1", "s2", "s3"])


def test_matching_sample_set(set_coords):
    assert set_coords.matching_sample_set() == "ALL"
    assert set_coords.with_sample_sets(["pop2"]).matching_sample_set() == "pop2"
    assert set_coords.with_samples(["s4", "s5"]).matching_sample_set() == "pop2"
    assert set_coords.with_samples(["s1"]).matching_sample_set() is None


def test_user_sample_set_names_follow_selection(set_coords):
    assert set_coords.user_sample_set_names == ["pop1", "pop2", "pop3"]
    assert set_coords.with_sample_sets(["pop2"]).user_sample_set_names == ["pop2"]


def test_sample_set_summary(set_coords):
    s = set_coords.with_sample_sets(["pop1"]).sample_set_summary()
    assert s.loc["ALL"].tolist() == [3, 7]
    assert s.loc["pop1"].tolist() == [3, 3]
    assert s.loc["pop2"].tolist() == [0, 2]


def test_overlapping_sample_sets(make_datastore, sample_sets):
    ss = sample_sets.copy(deep=True)
    ss[0, 1] = True  # s1 in pop1 and pop2
    coord = Coordinates(make_datastore(sample_sets=ss))
    assert coord.with_sample_sets(["pop1"]).samples.tolist() == ["s1", "s2", "s3"]
    assert coord.with_sample_sets(["pop2"]).samples.tolist() == ["s1", "s4", "s5"]
