from dataclasses import dataclass

import numpy as np
import pytest

from pbview.model.coordinates import Coordinates
from pbview.model.datastore import DataStore


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
def ds(store, sampleinfo, sample_sets):
    if sample_sets is None:
        return DataStore(path=store)
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture
def coordinate_args(ds):
    return ds, None


@pytest.fixture
def coordinates(coordinate_args):
    ds, param = coordinate_args
    return Coordinates(datastore=ds), param


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


def test_with_length_filter(coordinates):
    coord, _ = coordinates
    assert coord.with_length_filter(lower=850_000).n_contigs == 2
    assert coord.with_length_filter(lower=850_000, upper=950_000).n_contigs == 1
    assert coord.with_length_filter(upper=950_000).n_contigs == 2


def test_with_contigs(coordinates):
    coord, _ = coordinates
    assert coord.with_contigs(["chr1", "chr3"]).n_contigs == 2
    assert coord.with_contigs(["chr2"]).n_contigs == 1
    assert coord.with_contigs(["chr4"]).n_contigs == 0
    assert coord.with_contigs(["chr1"]).with_contigs().n_contigs == 3


def test_with_contigs_and_length_filters(coordinates):
    coord, _ = coordinates
    assert coord.with_contigs(["chr1"]).with_length_filter(lower=850_000).n_contigs == 1
    assert (
        coord.with_contigs(["chr1", "chr2"]).with_length_filter(lower=950_000).n_contigs
        == 1
    )


def test_with_samples(coordinates, expected_sample_set_membership_size):
    coord, _ = coordinates
    assert coord.with_samples(["s1", "s2"]).n_samples == 2
    assert coord.with_samples(["s1", "s8"]).n_samples == 1
    assert coord.with_samples(["s8"]).n_samples == 0
    assert coord.with_samples().n_samples == 7


def test_with_sample_sets(coordinates, expected_sample_set_membership_size):
    coord, _ = coordinates
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


def test_with_samples_and_sample_sets(coordinates, expected_sample_set_membership_size):
    coord, _ = coordinates
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


def test_contig_slices(coordinates):
    coord, _ = coordinates
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


def test_reset(coordinates):
    coord, _ = coordinates
    filtered_coord = coord.with_length_filter(lower=850_000).with_samples(["s1", "s2"])
    assert filtered_coord.n_contigs == 2
    assert filtered_coord.n_samples == 2
    reset_coord = filtered_coord.reset()
    assert reset_coord.n_contigs == 3
    assert reset_coord.n_samples == 7


def test_base_coord(coordinates):
    coord, _ = coordinates
    filtered_coord = coord.with_length_filter(lower=850_000).with_samples(["s1", "s2"])
    assert filtered_coord.n_contigs == 2
    assert filtered_coord.n_samples == 2
    assert filtered_coord.base_coord.n_contigs == 3
    assert filtered_coord.base_coord.n_samples == 7


def test_as_pyranges(coordinates):
    coord, _ = coordinates
    filtered_coord = coord.with_length_filter(lower=850_000).with_samples(["s1", "s2"])
    pr1 = coord.as_pyranges()
    pr2 = filtered_coord.as_pyranges()
    pr3 = filtered_coord.base_coord.as_pyranges()
    assert len(pr1.chromosomes) == 3
    assert len(pr2.chromosomes) == 2
    assert len(pr3.chromosomes) == 3
