import warnings
from dataclasses import dataclass

import _helpers
import numpy as np
import pyranges1 as pr
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
    return Genome(contigs=("chr1", "chr2", "chr10"), offsets=(0, 1000, 1800, 2400))


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
    assert len(pi) == expected


@pytest.mark.parametrize(
    "position_intervals,expected",
    [
        (((0, 100), (200, 300)), 200),
        ((()), 0),
    ],
)
def test_position_intervals_as_pyranges(position_intervals, expected):
    pi = PositionIntervals(position_intervals)
    ret = pi.as_pyranges()
    assert isinstance(ret, pr.PyRanges)
    assert ret.length == expected


def test_genome_init():
    genome = Genome(contigs=("chr10", "chr1"), offsets=(0, 1000, 1800))
    assert genome.contigs == ("chr10", "chr1")
    assert genome.offsets == (0, 1000, 1800)
    assert genome.size == 1800


def test_genome_init_raises():
    with pytest.raises(ValueError):
        _ = Genome(contigs=("chr1", "chr2"), offsets=(0, 1000))


def test_genome_as_pyranges():
    genome = Genome(contigs=("chr1", "chr2"), offsets=(0, 1000, 1800))
    ret = genome.as_pyranges()
    assert isinstance(ret, pr.PyRanges)
    assert ret.length == 1800


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


@pytest.mark.parametrize("contigs", [None, [], ()])
def test_spatial_selection_empty_is_noop(contigs):
    sel = SpatialSelection(contigs=contigs)
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
            (
                "chr1",
                "chr2",
                "chr10",
            ),
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
        ({"contigs": ("chr10",)}, (600, ((1800, 2400),))),
        (
            {"contigs": ("chr10", "chr1")},
            (
                1600,
                (
                    (0, 1000),
                    (1800, 2400),
                ),
            ),
        ),
    ],
)
def test_genome_resolve(genome, kw, expected):
    sel = SpatialSelection(**kw)
    pi = genome.resolve(sel)
    assert len(pi) == expected[0]
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


def test_genome_contig_slices(genome, coords):
    slices = genome.contig_slices(coords.with_contigs(["chr2"]))
    assert slices == ((1000, 1800),)
    slices = genome.contig_slices(coords.with_contigs(["chr1", "chr10"]))
    assert slices == (
        (0, 1000),
        (1800, 2400),
    )
    slices = genome.contig_slices(coords.with_contigs(["chr2", "chr10"]))
    assert slices == ((1000, 2400),)


def test_coordinates_with_contigs(coords, genome):
    ret = coords.with_contigs(["chr1", "chr2"])
    assert ret.n_contigs == 2


def test_coordinates_with_contigs_unknown_raises(coords):
    with pytest.raises(ValueError):
        coords.with_contigs(["chr1", "chr4"])


def test_coordinates_with_samples(coords):
    pass


def test_coordinates_with_sample_sets(coords):
    pass


def test_coordinates_with_length_filter(coords):
    pass


def test_with_length_filter(legacy_coords):
    coord = legacy_coords
    assert coord.with_length_filter(lower=850_000).n_contigs == 2
    assert coord.with_length_filter(lower=850_000, upper=950_000).n_contigs == 1
    assert coord.with_length_filter(upper=950_000).n_contigs == 2


# FIXME: Revise all tests to use small test fixtures, followed by
# transition to the new selection API.
@dataclass(frozen=True)
class ExpectedSampleSetsSize:
    default: int
    total_pop1: int
    n_sample_sets_pop1: int
    n_sampleset_membership_dataframe_pop1: int
    n_user_sample_sets_pop1: int
    n_samples_pop1: int
    n_samples_s1_s2_pop1: int


@pytest.fixture(params=["has_sample_sets", "no_sample_sets"])
def sample_sets(request):
    if request.param == "has_sample_sets":
        return np.array(["pop1", "pop1", "pop1", "pop2", "pop2", "pop3", "pop3"])
    return None


@pytest.fixture()
def ds(store, sampleset_info, sample_sets):
    if sample_sets is None:
        return DataStore(path=store)
    return DataStore(path=store, sampleset_info=sampleset_info)


@pytest.fixture
def coordinate_args(ds):
    return ds


@pytest.fixture
def legacy_coords(coordinate_args):
    ds = coordinate_args
    return Coordinates(datastore=ds)


@pytest.fixture
def expected_sample_set_membership_size(sample_sets):
    if sample_sets is None:
        return ExpectedSampleSetsSize(
            default=1,
            total_pop1=1,
            n_sampleset_membership_dataframe_pop1=0,
            n_sample_sets_pop1=1,
            n_user_sample_sets_pop1=0,
            n_samples_pop1=0,
            n_samples_s1_s2_pop1=0,
        )
    return ExpectedSampleSetsSize(
        default=3,
        total_pop1=2,
        n_sampleset_membership_dataframe_pop1=2,
        n_sample_sets_pop1=2,
        n_user_sample_sets_pop1=1,
        n_samples_pop1=3,
        n_samples_s1_s2_pop1=2,
    )


def test_with_contigs(legacy_coords):
    coord = legacy_coords
    assert coord.with_contigs(["chr1", "chr3"]).n_contigs == 2
    assert coord.with_contigs(["chr2"]).n_contigs == 1
    assert coord.with_contigs(["chr1"]).with_contigs().n_contigs == 3


def test_with_contigs_unknown_raises(legacy_coords):
    coord = legacy_coords
    with pytest.raises(ValueError):
        coord.with_contigs(["chr4"])


def test_with_contigs_and_length_filters(legacy_coords):
    coord = legacy_coords
    assert coord.with_contigs(["chr1"]).with_length_filter(lower=850_000).n_contigs == 1
    assert (
        coord.with_contigs(["chr1", "chr2"]).with_length_filter(lower=950_000).n_contigs
        == 1
    )


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


def test_with_samples(legacy_coords, expected_sample_set_membership_size):
    coord = legacy_coords
    assert coord.with_samples(["s1", "s2"]).n_samples == 2
    assert coord.with_samples(["s1", "s8"]).n_samples == 1
    assert coord.with_samples(["s8"]).n_samples == 0
    assert coord.with_samples().n_samples == 7


def test_with_sample_sets(legacy_coords, expected_sample_set_membership_size):
    coord = legacy_coords
    assert (
        coord.with_sample_sets(sample_sets=["pop1"]).n_user_sample_sets
        == expected_sample_set_membership_size.n_user_sample_sets_pop1
    )
    assert (
        coord.with_sample_sets(sample_sets=["pop1"]).n_sample_sets
        == expected_sample_set_membership_size.n_sample_sets_pop1
    )
    assert (
        coord.with_sample_sets(sample_sets=["pop1"]).n_samples
        == expected_sample_set_membership_size.n_samples_pop1
    )
    assert coord.with_sample_sets(sample_sets=["pop0"]).n_user_sample_sets == 0
    assert coord.with_sample_sets(sample_sets=["pop0"]).n_sample_sets == 1
    assert (
        len(
            set(
                coord.with_sample_sets(sample_sets=["pop1"])
                .sample_set_membership_dataframe(active_only=True)["sample_set"]
                .values
            )
        )
        == expected_sample_set_membership_size.n_sampleset_membership_dataframe_pop1
    )


def test_with_samples_and_sample_sets(
    legacy_coords, expected_sample_set_membership_size
):
    coord = legacy_coords
    assert (
        coord.with_samples(["s1", "s2"])
        .with_sample_sets(sample_sets=["pop2"])
        .n_samples
        == 0
    )
    assert (
        coord.with_samples(["s1", "s2"])
        .with_sample_sets(sample_sets=["pop1"])
        .n_samples
        == expected_sample_set_membership_size.n_samples_s1_s2_pop1
    )


def test_contig_slices(legacy_coords):
    coord = legacy_coords
    filtered_coord = coord.with_contigs(["chr1", "chr2"])
    assert len(filtered_coord.contig_slices()) == 1
    assert filtered_coord.contig_slices()[0][0] == 0
    assert filtered_coord.contig_slices()[0][1] == 1_900_000
    filtered_coord = coord.with_contigs(["chr1", "chr3"])
    assert len(filtered_coord.contig_slices()) == 2
    assert filtered_coord.contig_slices()[0][0] == 0
    assert filtered_coord.contig_slices()[0][1] == 1_000_000
    assert filtered_coord.contig_slices()[1][0] == 1_900_000
    assert filtered_coord.contig_slices()[1][1] == 2_700_000


def test_reset(legacy_coords):
    coord = legacy_coords
    filtered_coord = coord.with_length_filter(lower=850_000).with_samples(["s1", "s2"])
    assert filtered_coord.n_contigs == 2
    assert filtered_coord.n_samples == 2
    reset_coord = filtered_coord.reset()
    assert reset_coord.n_contigs == 3
    assert reset_coord.n_samples == 7


def test_base_coord(legacy_coords):
    coord = legacy_coords
    filtered_coord = coord.with_length_filter(lower=850_000).with_samples(["s1", "s2"])
    assert filtered_coord.n_contigs == 2
    assert filtered_coord.n_samples == 2
    assert filtered_coord.base_coord.n_contigs == 3
    assert filtered_coord.base_coord.n_samples == 7


def test_as_pyranges(legacy_coords):
    coord = legacy_coords
    filtered_coord = coord.with_length_filter(lower=850_000).with_samples(["s1", "s2"])
    pr1 = coord.as_pyranges()
    pr2 = filtered_coord.as_pyranges()
    pr3 = filtered_coord.base_coord.as_pyranges()
    assert len(pr1.chromosomes) == 3
    assert len(pr2.chromosomes) == 2
    assert len(pr3.chromosomes) == 3


# FIXME: add cache hit test
def test_legacy_coords_cache_hit(legacy_coords, request):
    coord = legacy_coords
    assert hash(coord) == hash(coord)
    assert hash(coord.with_sample_sets(["pop1"])) == hash(
        coord.with_sample_sets(["pop1"])
    )
    assert hash(coord) != hash(coord.with_sample_sets(["pop1"]))
