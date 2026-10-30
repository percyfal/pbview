import numpy as np
import pandas as pd
import pytest

from pbview import config
from pbview.model.grouping import (
    IDENTITY_DIM,
    ensure_all_set,
    grouping_sets_from_dataframe,
    identity_grouping,
    membership_from_sets,
    write_column_groupings,
)


@pytest.fixture
def sample_set_info():
    return pd.DataFrame(
        {
            "sample": ["s2", "s1", "s3", "s4", "s5", "s6", "s7", "s1", "s2", "s6"],
            "sample_set": [
                "pop1",
                "pop1",
                "pop1",
                "pop2",
                "pop2",
                "pop3",
                "pop3",
                "pop4",
                "pop4",
                "pop4",
            ],
        }
    )


@pytest.fixture
def group_data(sample_set_info):
    df = sample_set_info.copy()
    columns = df["sample"].unique().tolist()
    groups = df["sample_set"].unique().tolist()
    column_dim = "sample"
    group_dim = "sample_set"
    sets = grouping_sets_from_dataframe(df, column_dim=column_dim, group_dim=group_dim)
    return columns, groups, sets, column_dim, group_dim


def test_grouping_sets_from_dataframe(sample_set_info):
    sets = grouping_sets_from_dataframe(
        sample_set_info, column_dim="sample", group_dim="sample_set"
    )
    assert set(sets.keys()) == {"pop1", "pop2", "pop3", "pop4"}
    assert set(sets["pop1"]) == {"s1", "s2", "s3"}
    assert set(sets["pop2"]) == {"s4", "s5"}
    assert set(sets["pop3"]) == {"s6", "s7"}
    assert set(sets["pop4"]) == {"s1", "s2", "s6"}


def test_membership_from_sets(group_data):
    columns, groups, sets, column_dim, group_dim = group_data
    membership = membership_from_sets(
        columns, sets, column_dim=column_dim, group_dim=group_dim
    )
    assert np.array_equal(membership.dims, (column_dim, group_dim))
    assert np.array_equal(membership[column_dim].values, columns)
    assert np.array_equal(membership[group_dim].values, groups)
    assert np.array_equal(membership.values[:, 0], [True] * 3 + [False] * 4)
    assert np.array_equal(
        membership.values[:, 1], [False] * 3 + [True] * 2 + [False] * 2
    )
    assert np.array_equal(membership.values[:, 2], [False] * 5 + [True] * 2)
    assert np.array_equal(
        membership.values[:, 3], [True, True, False, False, False, True, False]
    )


def test_identity_grouping():
    g = identity_grouping(["a", "b"], "sample")
    assert g.dims == ("sample", IDENTITY_DIM)
    assert g[IDENTITY_DIM].values.tolist() == [config.DEFAULT_GROUPING_SET]
    assert bool(g.all())


def test_ensure_all_set(group_data):
    columns, groups, sets, column_dim, group_dim = group_data
    m = membership_from_sets(columns, sets, column_dim=column_dim, group_dim=group_dim)
    out = ensure_all_set(m)
    assert out[group_dim].values.tolist()[0] == config.DEFAULT_GROUPING_SET
    assert bool(out.sel({group_dim: config.DEFAULT_GROUPING_SET}).all())
    assert ensure_all_set(out).equals(out)  # idempotent


def test_ensure_all_set_rejects_partial_all(group_data):
    columns, _, _, column_dim, group_dim = group_data
    m = membership_from_sets(
        columns, {"ALL": columns[:1]}, column_dim=column_dim, group_dim=group_dim
    )
    with pytest.raises(ValueError):
        ensure_all_set(m)


def test_write_rejects_identity_dim(tmp_path):
    g = identity_grouping(["a", "b"], "sample")
    with pytest.raises(ValueError):
        write_column_groupings(tmp_path / "s.zarr", "depth", [g])
