import dask
import pytest

from pbview import config
from pbview.model.coordinates import Coordinates
from pbview.model.histogram import compute_threshold_defaults
from pbview.model.thresholds import profile_from_defaults, profile_from_state
from pbview.view.coordinates import selection_overview
from pbview.view.datastore import DataStoreView
from pbview.view.track import estimate_label, estimate_note, selection_stride, titled

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


def test_pages_read_a_partial_sample_selection_in_one_pass(full_ds, monkeypatch):
    dsv = DataStoreView(full_ds, max_positions=0)
    dsv.state.samples = ["s1", "s4", "s5", "s6"]  # not a sample set
    calls = []
    real = dask.compute
    monkeypatch.setattr(
        dask, "compute", lambda *a, **k: calls.append(len(a)) or real(*a, **k)
    )
    cov, mis = dsv.pages["Coverage"], dsv.pages["Missingness"]
    coverage = {s: rx.rx.value for s, rx in cov._hist_rx.items()}
    missingness = {s: rx.rx.value for s, rx in mis._hist_rx.items()}
    # the coverage page reads the summary sets (4) and the raw sets once, with
    # their missingness (3 raw selections x (coverage + cutoffs)); the
    # missingness page then reads only its summary set
    raw_graphs = 3 * (1 + len(full_ds.tracks["depth"].summaries.missingness))
    assert sorted(calls) == sorted([4, raw_graphs, 4])
    track = dsv.state.track()
    for s in SETS:
        resolved = dsv.state.coord.with_sample_sets([s]).resolve_for(
            track, full_ds.genome
        )
        counts, _ = track.coverage_hist(coverage[s][1].astype(int), resolved)
        assert (coverage[s][0] == counts).all()
        counts, _ = track.missingness_hist(resolved, dsv.state.missing_cutoff)
        assert (missingness[s][0] == counts).all()


def test_estimate_label_and_title():
    assert estimate_label(1) == ""
    assert estimate_label(14) == "estimated from every 14th position"
    assert titled("Coverage", 1) == "Coverage"
    assert titled("Coverage", 14) == "Coverage (estimated from every 14th position)"


def test_selection_stride(full_ds):
    dsv = DataStoreView(full_ds)
    state, track = dsv.state, dsv.state.track()
    assert selection_stride(state.coord, state, track, 0) == 1
    assert selection_stride(state.coord, state, track, 24) == 1
    assert selection_stride(state.coord, state, track, 5) == 5
    chr2 = state.coord.with_contigs(["chr2"])
    n = chr2.resolve_for(track, full_ds.genome).intervals.n_positions
    assert selection_stride(chr2, state, track, 5) == -(-n // 5)


def test_tables_say_when_values_are_estimates(full_ds):
    exact = DataStoreView(full_ds, max_positions=0)
    est = DataStoreView(full_ds, max_positions=5)
    for name in ("Coverage", "Missingness"):
        assert "sampling" not in exact.pages[name].table._df_rx()
        df = est.pages[name].table._df_rx()
        assert set(df["sampling"]) == {"estimated from every 5th position"}
        assert len(df) == len(SETS)


def test_plots_say_when_values_are_estimates(full_ds):
    exact = DataStoreView(full_ds, max_positions=0)
    est = DataStoreView(full_ds, max_positions=5)
    for dsv, suffix in ((exact, ""), (est, " (estimated from every 5th position)")):
        for name in ("Coverage", "Missingness"):
            page = dsv.pages[name]
            for s, view in page._plots.items():
                assert view._title(dsv.state.coord, s) == s + suffix
        # the plots still render with the new argument
        cov, mis = dsv.pages["Coverage"], dsv.pages["Missingness"]
        cov._plots["ALL"]._plot(
            cov._hist_rx["ALL"].rx.value, 10, 50, False, dsv.state.coord
        )
        mis._plots["ALL"]._plot(mis._hist_rx["ALL"].rx.value, 1, dsv.state.coord)
        cov._make_boxplot(*[rx.rx.value for rx in cov._hist_rx.values()])
