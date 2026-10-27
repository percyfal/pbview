import warnings

import _helpers
import numpy as np
import pytest

from pbview._compat import ShimWarning
from pbview.model.coordinates import (
    ColumnSelection,
    Coordinates,
    Genome,
    PositionIntervals,
    SpatialSelection,
)
from pbview.model.datastore import DataStore


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
    sel = ColumnSelection(column_dim="sample", selected=selected)
    ret = sel.apply(data)
    assert ret == data


@pytest.mark.parametrize("selected", [None, [], ()])
def test_empty_column_selection_on_subset_is_noop(data, selected):
    data = data.copy()
    data = data.sel(sample=["s1"])
    ret = ColumnSelection(column_dim="sample", selected=selected).apply(data)
    assert ret == data


def test_column_selection_selected(data):
    sel = ColumnSelection(column_dim="sample", selected=["s1", "s3"])
    ret = sel.apply(data)
    assert ret.coords["sample"].values.tolist() == ["s1", "s3"]


@pytest.mark.skip("Coordinates will not know about sample sets going forward")
@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, 0),
        ({"with_all": True}, 1),
        ({"with_all": True, "sample_sets": _helpers.sample_sets()}, 3),
    ],
)
def test_coordinates_init(make_datastore, kw, expected):
    ds = make_datastore(**kw)
    coords = Coordinates(ds)
    assert coords.n_sample_sets == expected


def test_coordinates_with_contigs(coords, genome):
    assert len(coords.with_contigs(["chr1", "chr2"]).selected_contigs(genome)) == 2
    assert len(coords.with_contigs(["chr2"]).selected_contigs(genome)) == 1
    assert (
        len(coords.with_contigs(["chr2"]).with_contigs().selected_contigs(genome)) == 3
    )


def test_coordinates_with_contigs_unknown_raises(coords):
    with pytest.raises(KeyError):
        coords.with_contigs(["chr1", "chr4"])


def test_coordinates_with_samples(coords):
    pass


def test_coordinates_with_sample_sets(coords):
    pass


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


@pytest.mark.parametrize(
    "contigs, expected",
    [
        (None, [True, True, True]),
        ((), [False, False, False]),
        (("chr10", "chr1"), [True, False, True]),
    ],
)
def test_post_init_derives_mask_from_spatial(legacy_coords, contigs, expected):  # SHIM
    coords = legacy_coords.with_contigs(contigs)
    np.testing.assert_array_equal(coords._contig_selected, expected)


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


def test_legacy_contig_mask_read_warns(legacy_coords):
    legacy_coords = legacy_coords
    with pytest.warns(ShimWarning, match="contig_selected"):
        legacy_coords.contig_selected


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
    assert coord.with_samples(["s1", "s8"]).n_samples == 1
    assert coord.with_samples(["s8"]).n_samples == 0
    assert coord.with_samples().n_samples == 7


# FIXME: update for new sample selection functions
def test_with_sample_sets(legacy_coords):
    _ = legacy_coords
    # assert (
    #     coord.with_sample_sets(sample_sets=["pop1"]).n_user_sample_sets
    #     == expected_sample_set_membership_size.n_user_sample_sets_pop1
    # )
    # assert (
    #     coord.with_sample_sets(sample_sets=["pop1"]).n_sample_sets
    #     == expected_sample_set_membership_size.n_sample_sets_pop1
    # )
    # assert (
    #     coord.with_sample_sets(sample_sets=["pop1"]).n_samples
    #     == expected_sample_set_membership_size.n_samples_pop1
    # )
    # assert coord.with_sample_sets(sample_sets=["pop0"]).n_user_sample_sets == 0
    # assert coord.with_sample_sets(sample_sets=["pop0"]).n_sample_sets == 1
    # assert (
    #     len(
    #         set(
    #             coord.with_sample_sets(sample_sets=["pop1"])
    #             .sample_set_membership_dataframe(active_only=True)["sample_set"]
    #             .values
    #         )
    #     )
    #     == expected_sample_set_membership_size.n_sampleset_membership_dataframe_pop1
    # )


# FIXME: update for new sample selection functions
def test_with_samples_and_sample_sets(legacy_coords):
    _ = legacy_coords
    # assert (
    #     coord.with_samples(["s1", "s2"])
    #     .with_sample_sets(sample_sets=["pop2"])
    #     .n_samples
    #     == 0
    # )
    # assert (
    #     coord.with_samples(["s1", "s2"])
    #     .with_sample_sets(sample_sets=["pop1"])
    #     .n_samples
    #     == expected_sample_set_membership_size.n_samples_s1_s2_pop1
    # )


def test_reset_clears_contigs(coords, genome):
    assert (
        coords.with_contigs(("chr1",)).reset().selected_contigs(genome)
        == genome.contigs
    )


def test_reset_from_empty_selection(coords, genome):
    assert coords.with_contigs(()).reset().selected_contigs(genome) == genome.contigs


def test_reset_is_default(coords):
    assert coords.with_contigs(("chr1",)).reset().spatial == SpatialSelection()


# FIXME: add cache hit test
def test_legacy_coords_cache_hit(legacy_coords, request):
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
