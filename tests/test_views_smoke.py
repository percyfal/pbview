import pytest

from pbview import config
from pbview.model.coordinates import Coordinates
from pbview.model.histogram import compute_threshold_defaults
from pbview.model.thresholds import profile_from_defaults, profile_from_state
from pbview.view.coordinates import selection_overview
from pbview.view.datastore import DataStoreView
from pbview.view.track import estimate_note

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


def test_selection_overview(full_ds):
    coords = Coordinates().with_contigs(["chr1"]).with_sample_sets(["pop1"])
    df = selection_overview(coords, full_ds.genome, full_ds.tracks["depth"])
    assert df.index.names == ["Type", "Label"]
    assert list(df.columns) == ["Active", "Total"]
    assert df.loc[("Coordinates", "Contigs")].tolist() == [1, 3]
    assert df.loc[("Coordinates", "Samples")].tolist() == [3, 7]
    assert df.loc["Sample sets"].index.tolist() == SETS
    assert df.loc[("Sample sets", "pop2")].tolist() == [0, 2]


def test_selection_overview_pane(dsv):
    table = dsv.cv.summary_pane()
    assert len(table.value) == 2 + len(SETS)


def test_summary_scatter_with_contig_subset(dsv):
    dsv.state.contigs = ["chr2"]
    df = dsv.pages["Summary"]._contig_scatter_df()
    assert len(df) == 3 * len(SETS)  # one row per contig and set
    emphasis = df.drop_duplicates("contig").set_index("contig")["selected"]
    assert emphasis["chr2"] > emphasis["chr1"]
    assert emphasis["chr2"] > emphasis["chr10"]


def test_summary_scatter_click_selects_contigs(dsv):
    page, state = dsv.pages["Summary"], dsv.state
    genome = state.datastore.genome
    rows = [i for i, c in enumerate(page._source.data["contig"]) if c == "chr1"]
    page._on_select("indices", [], rows[:1])
    assert state.contigs == ["chr1"]
    assert state.coord.selected_contigs(genome) == ("chr1",)
    page._on_select("indices", rows[:1], [])  # clearing the selection selects all
    assert set(state.contigs) == {"chr1", "chr2", "chr10"}


def test_exported_thresholds_match_the_app_defaults(full_ds, dsv):
    defaults = compute_threshold_defaults(full_ds, "depth")
    exported = profile_from_defaults(
        full_ds.tracks["depth"], defaults, full_ds.id, missing_cutoff=3
    )
    assert exported == profile_from_state(dsv.state, full_ds.id)


def test_app_defaults_honour_max_positions(full_ds):
    dsv = DataStoreView(full_ds, max_positions=5)
    defaults = compute_threshold_defaults(full_ds, "depth", max_positions=5)
    assert defaults != compute_threshold_defaults(full_ds, "depth", max_positions=0)
    expected = profile_from_defaults(
        full_ds.tracks["depth"], defaults, full_ds.id, missing_cutoff=3
    )
    assert profile_from_state(dsv.state, full_ds.id) == expected


def test_pages_estimate_when_there_are_more_positions_than_max_positions(full_ds):
    dsv = DataStoreView(full_ds, max_positions=5)  # 24 positions, every 5th
    assert dsv.pages["Coverage"].max_positions == 5
    assert dsv.pages["Missingness"].max_positions == 5
    for page in (dsv.pages["Coverage"], dsv.pages["Missingness"]):
        for rx in page._hist_rx.values():
            counts, _ = rx.rx.value
            assert abs(counts.sum() - 24) <= 5  # scaled up to the positions
    assert (
        DataStoreView(full_ds).pages["Coverage"].max_positions == config.MAX_POSITIONS
    )


def test_estimate_note(full_ds):
    dsv = DataStoreView(full_ds, max_positions=5)
    track = dsv.state.track()
    note = estimate_note(dsv.state.coord, dsv.state, track, 5)
    assert "every 5th of 24 selected positions" in note.object
    assert estimate_note(dsv.state.coord, dsv.state, track, 0).object == ""
    assert estimate_note(dsv.state.coord, dsv.state, track, 24).object == ""
