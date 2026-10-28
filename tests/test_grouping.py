import numpy as np
import pandas as pd
import pytest

from pbview.model.grouping import grouping_sets_from_dataframe, membership_from_sets


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
