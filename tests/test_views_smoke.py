import dask
import panel as pn
import pytest

from pbview import config
from pbview.model.coordinates import Coordinates
from pbview.model.histogram import compute_threshold_defaults
from pbview.model.thresholds import profile_from_defaults, profile_from_state
from pbview.view.coordinates import (
    active_genome_size,
    genome_size_gauges,
    selection_overview,
)
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


def test_active_genome_size(full_ds):
    genome = full_ds.genome
    everything = Coordinates()
    assert active_genome_size(everything, genome) == genome.size
    chr2 = everything.with_contigs(["chr2"])
    expected = int(genome.lengths[genome.contig_indices(["chr2"])].sum())
    assert 0 < expected < genome.size
    assert active_genome_size(chr2, genome) == expected
    assert active_genome_size(everything.with_contigs([]), genome) == 0


def test_genome_gauges_show_the_active_contigs(full_ds):
    genome = full_ds.genome
    size, fraction = genome_size_gauges(Coordinates(), genome)
    assert (size.value, fraction.value) == (genome.size, 100.0)
    subset = Coordinates().with_contigs(["chr1", "chr2"])
    active = active_genome_size(subset, genome)
    size, fraction = genome_size_gauges(subset, genome)
    assert size.value == active
    assert size.bounds == (0, genome.size)  # the whole genome is the scale
    assert fraction.value == round(active / genome.size * 100, 2)
    assert "Active" in size.label


def test_genome_gauges_follow_the_selection(full_ds):
    dsv = DataStoreView(full_ds, max_positions=0)
    genome = full_ds.genome
    pane = pn.panel(dsv.cv.genome_size_pane)  # as the sidebar uses it
    assert pane._pane[0].value == genome.size
    dsv.state.contigs = ["chr2"]
    assert pane._pane[0].value == active_genome_size(dsv.state.coord, genome)
    assert pane._pane[0].value < genome.size
    dsv.state.lower = int(genome.lengths.max())  # the length filter counts too
    assert pane._pane[0].value == active_genome_size(dsv.state.coord, genome)


def test_pca_page_is_in_the_app(dsv):
    assert list(dsv.pages) == ["Summary", "Coverage", "Missingness", "PCA"]
    assert "PCA" in dsv.param.active_page.objects


def test_pca_page_waits_for_the_button(dsv):
    page = dsv.pages["PCA"]
    assert page.result is None
    status = page._status(None, None, dsv.state.coord, page.window_size)
    assert "Run PCA" in status.object
    assert page._scatter(None, 1, 2, False).object == ""


def test_pca_page_runs_for_the_selection(dsv):
    page = dsv.pages["PCA"]
    page.window_size = 3
    page.run()
    result = page.result
    assert result.n_windows == 7 and len(result.samples) == 7
    assert page.param.x_component.objects == list(range(1, 8))
    assert (page.x_component, page.y_component) == (1, 2)
    status = page._status(result, page.ran_for, dsv.state.coord, 3).object
    assert "PCA of 7 samples in 7 windows of 3 bp" in status
    # the same result is returned for the same selection, not computed again
    page.run()
    assert page.result is result


def test_pca_page_draws_the_samples_by_sample_set(dsv):
    page = dsv.pages["PCA"]
    page.window_size = 3
    page.run()
    plot = page._scatter(page.result, 1, 2, False)
    assert [e.label for e in plot] == ["pop1", "pop2", "pop3"]
    assert sum(len(e) for e in plot) == 7  # every sample once
    with_labels = page._scatter(page.result, 1, 3, True)
    assert len(with_labels) == 4  # three sets and the labels
    assert page._scatter(page.result, 1, 20, False).object == ""
    assert len(page._scree(page.result)) == 7
    assert len(page._table(page.result).value) == 7


def test_pca_page_says_when_the_selection_has_changed(dsv):
    page = dsv.pages["PCA"]
    page.window_size = 3
    page.run()
    dsv.state.contigs = ["chr1", "chr10"]
    status = page._status(page.result, page.ran_for, dsv.state.coord, 3).object
    assert "changed since the last run" in status
    page.run()
    assert page.result.n_windows == 5  # 3 + 2
    status = page._status(page.result, page.ran_for, dsv.state.coord, 3).object
    assert "PCA of 7 samples in 5 windows" in status
    page.window_size = 4
    status = page._status(page.result, page.ran_for, dsv.state.coord, 4).object
    assert "changed since the last run" in status


def test_pca_page_reports_an_error_instead_of_failing(dsv):
    page = dsv.pages["PCA"]
    page.window_size = 11  # no contig is that long
    page.run()
    assert page.result is None
    status = page._status(None, page.ran_for, dsv.state.coord, 11).object
    assert "Not computed" in status and "at least two windows" in status


def test_pca_page_scores_can_be_downloaded(dsv):
    page = dsv.pages["PCA"]
    assert page._scores_tsv().read() == ""  # nothing yet
    page.window_size = 3
    page.run()
    lines = page._scores_tsv().read().splitlines()
    assert lines[0].split("\t")[:3] == ["sample", "sample_sets", "PC1"]
    assert len(lines) == 8  # header and seven samples
    assert lines[1].split("\t")[:2] == ["s1", "pop1"]


def test_pca_page_renders(dsv):
    # the layout builds with and without a result
    page = dsv.pages["PCA"]
    pn.panel(page.__panel__())
    page.window_size = 3
    page.run()
    pn.panel(page.__panel__())
