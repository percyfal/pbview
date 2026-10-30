import pytest

from pbview.view.datastore import DataStoreView

SETS = ["ALL", "pop1", "pop2", "pop3"]


@pytest.fixture()
def dsv(full_ds):
    return DataStoreView(full_ds)


def test_coverage_page(dsv):
    page = dsv.pages["Coverage"]
    assert list(page._hist_rx) == SETS
    for rx in page._hist_rx.values():
        counts, bins = rx.rx.value
        assert counts.sum() > 0
    assert page.table._df_rx()["sample_set"].tolist() == SETS
    page.by_sample = True
    for rx in page._hist_rx.values():
        rx.rx.value
    assert len(page.table._df_rx()) == len(SETS)


def test_missingness_page(dsv):
    page = dsv.pages["Missingness"]
    assert list(page._hist_rx) == SETS
    for rx in page._hist_rx.values():
        counts, bins = rx.rx.value
        assert counts.sum() > 0
    assert page.table._df_rx()["sample_set"].tolist() == SETS


def test_summary_page(dsv):
    df = dsv.pages["Summary"]._contig_scatter_df()
    assert sorted(df["sample_set"].unique()) == sorted(SETS)
    assert len(df) == 3 * len(SETS)
