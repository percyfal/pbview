import numpy as np
import pytest

from pbview.datastore import DataStore
from pbview.view import SelectionState


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def state(ds):
    return SelectionState(datastore=ds)


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"lower": 950_000}, {"size": 1_000_000, "contigs_size": 1}),
        (
            {"upper": 950_000},
            {"size": 1_700_000, "contigs_size": 2},
        ),
        ({"lower": 850_000, "upper": 950_000}, {"size": 900_000, "contigs_size": 1}),
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
        ({"contigs": []}, {"size": 2_700_000, "contigs_size": 3}),
        ({"contigs": ["chr1"]}, {"size": 1_000_000, "contigs_size": 1}),
        ({"contigs": ["chr2"]}, {"size": 900_000, "contigs_size": 1}),
        ({"contigs": ["chr1", "chr2"]}, {"size": 1_900_000, "contigs_size": 2}),
        ({"contigs": ["chr4"]}, {"size": 0, "contigs_size": 0}),
    ],
)
def test_contigs_filter(state, args, expected):
    state.contigs = args["contigs"]
    assert state.coord.size == expected["size"]
    assert state.coord.contigs_size == expected["contigs_size"]


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {"contigs": [], "lower": 0, "upper": np.inf},
            {"size": 2_700_000, "contigs_size": 3},
        ),
        (
            {"contigs": ["chr1", "chr2"], "lower": 950_000, "upper": np.inf},
            {"size": 1_000_000, "contigs_size": 1},
        ),
        (
            {"contigs": ["chr2", "chr3"], "lower": 0, "upper": 850_000},
            {"size": 800_000, "contigs_size": 1},
        ),
        (
            {"contigs": [], "lower": 850_000, "upper": 950_000},
            {"size": 900_000, "contigs_size": 1},
        ),
    ],
)
def test_contigs_and_length_filters(state, args, expected):
    state.contigs = args["contigs"]
    state.lower = args["lower"]
    state.upper = args["upper"]
    assert state.coord.size == expected["size"]
    assert state.coord.contigs_size == expected["contigs_size"]
