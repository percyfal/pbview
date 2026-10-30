import numpy as np
import pytest

from pbview.model.coordinates import (
    ColumnSelection,
    Coordinates,
    Genome,
    SpatialSelection,
)
from pbview.model.selection import SelectionStateBase, make_selection_state_class


@pytest.fixture()
def genome():
    return Genome(
        contigs=["chr1", "chr2", "chr10"],
        offsets=[0, 10, 18, 24],
    )


@pytest.fixture()
def coords(raw_ds):
    return Coordinates(raw_ds)


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
def state(full_ds, defaults):
    coord = Coordinates(full_ds)
    # full_ds.missing_cutoffs = [3]  # FIXME: Hack to avoid missing_cutoffs
    cls = make_selection_state_class(full_ds.tracks["depth"], coord, defaults=defaults)
    return cls(datastore=full_ds)


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"lower": 9}, {"genome_size": 10, "n_contigs": 1}),
        (
            {"upper": 9},
            {"genome_size": 14, "n_contigs": 2},
        ),
        (
            {"lower": 7, "upper": 9},
            {"genome_size": 8, "n_contigs": 1},
        ),
    ],
)
def test_length_filters(state, args, expected):
    genome = state.datastore.genome
    for k, v in args.items():
        setattr(state, k, v)
    pi = genome.resolve(state.coord.spatial)
    for k, v in expected.items():
        if k == "genome_size":
            assert pi.n_positions == v
        elif k == "n_contigs":
            assert len(genome.resolve_contigs(state.coord.spatial)) == v
        else:
            assert getattr(state.coord, k) == v


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"contigs": []}, {"genome_size": 24, "n_contigs": 3}),
        ({"contigs": ["chr1"]}, {"genome_size": 10, "n_contigs": 1}),
        ({"contigs": ["chr2"]}, {"genome_size": 8, "n_contigs": 1}),
        ({"contigs": ["chr1", "chr2"]}, {"genome_size": 18, "n_contigs": 2}),
    ],
)
def test_contigs_filter(state, args, expected, genome):
    state.contigs = args["contigs"]
    assert genome.resolve(state.coord.spatial).n_positions == expected["genome_size"]
    assert len(genome.resolve_contigs(state.coord.spatial)) == expected["n_contigs"]


def test_contigs_filter_unknown_raises(state):
    with pytest.raises(KeyError):
        state.contigs = ["chr4"]


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {"contigs": [], "lower": 0, "upper": np.inf},
            {"genome_size": 24, "n_contigs": 3},
        ),
        (
            {"contigs": ["chr1", "chr2"], "lower": 9, "upper": np.inf},
            {"genome_size": 10, "n_contigs": 1},
        ),
        (
            {"contigs": ["chr2", "chr10"], "lower": 0, "upper": 7},
            {"genome_size": 6, "n_contigs": 1},
        ),
        (
            {"contigs": ["chr10", "chr2", "chr1"], "lower": 7, "upper": 9},
            {"genome_size": 8, "n_contigs": 1},
        ),
    ],
)
def test_contigs_and_length_filters(state, args, expected):
    state.contigs = args["contigs"]
    state.lower = args["lower"]
    state.upper = args["upper"]
    pi = state.datastore.genome.resolve(state.coord.spatial)
    assert pi.n_positions == expected["genome_size"]
    assert (
        len(state.coord.datastore.genome.resolve_contigs(state.coord.spatial))
        == expected["n_contigs"]
    )


def test_filter_with_empty_picker_applies_length(state):
    state.contigs, state.lower = [], 9
    assert state._current().spatial.contigs == ("chr1",)  # not None/all


def test_filter_nothing_passes_is_empty(state):
    state.contigs, state.lower = [], 10**9
    assert state._current().spatial == SpatialSelection(())


def test_filtered_applies_columns(coords):
    sel = ColumnSelection("sample", ("s1",))
    c = SelectionStateBase._filtered(coords, 0, float("inf"), (sel,), ())
    assert c.column == (sel,)


def test_state_requires_summaries(raw_ds):
    with pytest.raises(ValueError, match="pbview preprocess"):
        SelectionStateBase(datastore=raw_ds)
