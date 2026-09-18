import numpy as np
import pytest

from pbview.model.datastore import DataStore
from pbview.model.selection import make_selection_state_class


@pytest.fixture()
def defaults():
    return {
        "lower_coverage_ALL": 10,
        "upper_coverage_ALL": 100,
        "lower_coverage_pop1": 3,
        "upper_coverage_pop1": 40,
        "lower_coverage_pop2": 5,
        "upper_coverage_pop2": 50,
        "lower_coverage_pop3": 7,
        "upper_coverage_pop3": 60,
    }


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def state(ds, defaults):
    cls = make_selection_state_class(ds.base_coord, defaults=defaults)
    return cls(datastore=ds)


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"lower": 950_000}, {"genome_size": 1_000_000, "n_contigs": 1}),
        (
            {"upper": 950_000},
            {"genome_size": 1_700_000, "n_contigs": 2},
        ),
        (
            {"lower": 850_000, "upper": 950_000},
            {"genome_size": 900_000, "n_contigs": 1},
        ),
    ],
)
def test_length_filters(state, args, expected):
    for k, v in args.items():
        setattr(state, k, v)
    for k, v in expected.items():
        assert getattr(state.coord, k) == v


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"contigs": []}, {"genome_size": 2_700_000, "n_contigs": 3}),
        ({"contigs": ["chr1"]}, {"genome_size": 1_000_000, "n_contigs": 1}),
        ({"contigs": ["chr2"]}, {"genome_size": 900_000, "n_contigs": 1}),
        ({"contigs": ["chr1", "chr2"]}, {"genome_size": 1_900_000, "n_contigs": 2}),
        ({"contigs": ["chr4"]}, {"genome_size": 0, "n_contigs": 0}),
    ],
)
def test_contigs_filter(state, args, expected):
    state.contigs = args["contigs"]
    assert state.coord.genome_size == expected["genome_size"]
    assert state.coord.n_contigs == expected["n_contigs"]


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {"contigs": [], "lower": 0, "upper": np.inf},
            {"genome_size": 2_700_000, "n_contigs": 3},
        ),
        (
            {"contigs": ["chr1", "chr2"], "lower": 950_000, "upper": np.inf},
            {"genome_size": 1_000_000, "n_contigs": 1},
        ),
        (
            {"contigs": ["chr2", "chr3"], "lower": 0, "upper": 850_000},
            {"genome_size": 800_000, "n_contigs": 1},
        ),
        (
            {"contigs": [], "lower": 850_000, "upper": 950_000},
            {"genome_size": 900_000, "n_contigs": 1},
        ),
    ],
)
def test_contigs_and_length_filters(state, args, expected):
    state.contigs = args["contigs"]
    state.lower = args["lower"]
    state.upper = args["upper"]
    assert state.coord.genome_size == expected["genome_size"]
    assert state.coord.n_contigs == expected["n_contigs"]
