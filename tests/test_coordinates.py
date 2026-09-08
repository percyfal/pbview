from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from pbview.datastore import Coordinates


@dataclass(frozen=True)
class ExpectedSampleSetsSize:
    default: int
    sample_sets_size_pop1: int
    samples_size_pop1: int


@pytest.fixture(params=["vectors", "DataTree"])
def coordinate_args(request):
    samples = np.array([f"s{i}" for i in np.arange(1, 8)])
    contigs = np.array(["chr1", "chr2", "chr3"])
    offsets = np.array([0, 1_000_000, 1_900_000, 2_700_000])
    if request.param == "vectors":
        return {
            "samples": samples,
            "contigs": contigs,
            "offsets": offsets,
        }, request.param
    elif request.param == "DataTree":
        tree = xr.DataTree.from_dict(
            {
                "/": xr.Dataset(
                    data_vars={
                        "contigs": ("contigs", contigs),
                        "offsets": ("offsets", offsets),
                    },
                    coords={"sample": samples},
                )
            }
        )
        return {"group": tree}, request.param


@pytest.fixture(params=["has_samplesets", "no_samplesets"])
def samplesets(request):
    if request.param == "has_samplesets":
        return np.array(["pop1", "pop1", "pop1", "pop2", "pop2", "pop3", "pop3"])
    return None


@pytest.fixture
def coordinates(coordinate_args, samplesets):
    args, param = coordinate_args
    if param == "DataTree":
        return Coordinates.from_datatree(
            group=args["group"], sample_sets=samplesets
        ), param
    return Coordinates(**args, sample_sets=samplesets), param


@pytest.fixture
def expected_sample_sets_size(samplesets):
    if samplesets is None:
        return ExpectedSampleSetsSize(
            default=1, sample_sets_size_pop1=0, samples_size_pop1=0
        )
    return ExpectedSampleSetsSize(
        default=3, sample_sets_size_pop1=1, samples_size_pop1=3
    )


def test_with_length_filter(coordinates):
    coord, _ = coordinates
    assert coord.with_length_filter(lower=850_000).contigs_size == 2
    assert coord.with_length_filter(lower=850_000, upper=950_000).contigs_size == 1
    assert coord.with_length_filter(upper=950_000).contigs_size == 2


def test_with_contigs(coordinates):
    coord, _ = coordinates
    assert coord.with_contigs(["chr1", "chr3"]).contigs_size == 2
    assert coord.with_contigs(["chr2"]).contigs_size == 1
    assert coord.with_contigs(["chr4"]).contigs_size == 0
    assert coord.with_contigs(["chr1"], lower=850_000).contigs_size == 1
    assert coord.with_contigs(["chr1", "chr2"], lower=950_000).contigs_size == 1
    assert coord.with_contigs(["chr1"]).with_contigs().contigs_size == 3


def test_with_samples(coordinates, expected_sample_sets_size):
    coord, _ = coordinates
    assert coord.with_samples(["s1", "s2"]).samples_size == 2
    assert coord.with_samples(["s1", "s8"]).samples_size == 1
    assert coord.with_samples(["s8"]).samples_size == 0
    assert coord.with_samples().samples_size == 7
    assert (
        coord.with_samples(sample_sets=["pop1"]).sample_sets_size
        == expected_sample_sets_size.sample_sets_size_pop1
    )
    assert (
        coord.with_samples(sample_sets=["pop1"]).samples_size
        == expected_sample_sets_size.samples_size_pop1
    )
    assert coord.with_samples(sample_sets=["pop0"]).sample_sets_size == 0


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
    assert filtered_coord.contigs_size == 2
    assert filtered_coord.samples_size == 2
    reset_coord = filtered_coord.reset()
    assert reset_coord.contigs_size == 3
    assert reset_coord.samples_size == 7
