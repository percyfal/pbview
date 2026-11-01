"""Behaviour tests for the command line."""

import copy
import io
import json
from pathlib import Path

import _helpers
import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner

import pbview.mask.export as _export
from pbview import config
from pbview.__main__ import cli
from pbview.model.coordinates import Coordinates
from pbview.model.datastore import DataStore
from pbview.model.histogram import compute_threshold_defaults
from pbview.model.thresholds import ThresholdProfile, profile_from_defaults

SAMPLESET_INFO = "s1 pop1\ns2 pop1\ns3 pop1\ns4 pop2\ns5 pop2\ns6 pop3\ns7 pop3\n"


@pytest.fixture(autouse=True)
def restore_dask_config():
    """The commands set the dask scheduler globally; undo that after each test."""
    saved = copy.deepcopy(dask.config.config)
    yield
    dask.config.config.clear()
    dask.config.config.update(saved)


@pytest.fixture
def invoke():
    runner = CliRunner()

    def _invoke(*args):
        return runner.invoke(cli, [str(a) for a in args])

    return _invoke


@pytest.fixture
def info(tmp_path):
    path = tmp_path / "sets.tsv"
    path.write_text(SAMPLESET_INFO)
    return path


def cutoffs(path):
    return sorted(DataStore(path).summaries_of("depth").missingness)


def test_preprocess_missingness_several_cutoffs(invoke, make_datastore_path, info):
    path = make_datastore_path()
    result = invoke(
        "preprocess",
        "missingness",
        path,
        "--sampleset-info",
        info,
        "--missing-data-threshold",
        3,
        "--missing-data-threshold",
        5,
        "--no-progress",
    )
    assert result.exit_code == 0, result.output
    assert cutoffs(path) == [3, 5]


def test_preprocess_all_defaults_to_cutoff_3(invoke, make_datastore_path, info):
    path = make_datastore_path()
    result = invoke(
        "preprocess", "all", path, "--sampleset-info", info, "--no-progress"
    )
    assert result.exit_code == 0, result.output
    assert cutoffs(path) == [3]


def test_preprocess_thresholds_needs_the_cutoff(
    invoke, make_datastore_path, sample_sets
):
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    result = invoke("preprocess", "thresholds", path, "--threshold", 9, "--no-progress")
    assert result.exit_code != 0
    assert "cutoff 9" in result.output
    assert "available: [3]" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("option", ["--missing-data-threshold", "--threshold"])
def test_preprocess_thresholds_accepts_both_spellings(
    invoke, make_datastore_path, sample_sets, option
):
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    result = invoke("preprocess", "thresholds", path, option, 3, "--no-progress")
    assert result.exit_code == 0, result.output


@pytest.fixture
def several_cutoff_mask(invoke, two_cutoff_store, threshold_file, tmp_path):
    output = tmp_path / "mask.zarr"
    result = invoke(
        "mask",
        "generate",
        two_cutoff_store,
        threshold_file,
        "-o",
        output,
        "--missing-data-threshold",
        3,
        "--missing-data-threshold",
        5,
        "--no-progress",
    )
    assert result.exit_code == 0, result.output
    return output


def test_mask_export_with_a_cutoff(invoke, several_cutoff_mask, tmp_path):
    bed = tmp_path / "ALL_5.bed"
    result = invoke(
        "mask",
        "export",
        several_cutoff_mask,
        "--sampleset-name",
        "ALL",
        "--missing-data-threshold",
        5,
        "-o",
        bed,
        "--no-progress",
    )
    assert result.exit_code == 0, result.output
    expected = io.StringIO()
    _export._export_mask_bed(
        _helpers.expected_accessible(5)["ALL"],
        np.array(["chr1", "chr2", "chr10"]),
        np.array([0, 10, 18, 24]),
        expected,
    )
    assert bed.read_text() == expected.getvalue()


def test_mask_export_without_a_cutoff_is_a_clean_error(
    invoke, several_cutoff_mask, tmp_path
):
    result = invoke(
        "mask", "export", several_cutoff_mask, "-o", tmp_path / "x.bed", "--no-progress"
    )
    assert result.exit_code != 0
    assert "--missing-data-threshold" in result.output
    assert "Traceback" not in result.output


def test_mask_generate_with_a_missing_summary_is_a_clean_error(
    invoke, two_cutoff_store, threshold_file, tmp_path
):
    result = invoke(
        "mask",
        "generate",
        two_cutoff_store,
        threshold_file,
        "-o",
        tmp_path / "m.zarr",
        "--missing-data-threshold",
        7,
        "--no-progress",
    )
    assert result.exit_code != 0
    assert "cutoff 7" in result.output
    assert "Traceback" not in result.output


@pytest.fixture
def preprocessed_store(make_datastore_path, sample_sets):
    return make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )


def test_preprocess_thresholds_writes_a_file(invoke, preprocessed_store, tmp_path):
    output = tmp_path / "thresholds.yml"
    result = invoke(
        "preprocess", "thresholds", preprocessed_store, "-o", output, "--no-progress"
    )
    assert result.exit_code == 0, result.output
    profile = ThresholdProfile.read(output)
    assert list(profile.sample_sets) == ["ALL", "pop1", "pop2", "pop3"]


def test_exported_thresholds_feed_mask_generate(invoke, preprocessed_store, tmp_path):
    thresholds = tmp_path / "thresholds.yml"
    result = invoke(
        "preprocess",
        "thresholds",
        preprocessed_store,
        "-o",
        thresholds,
        "--no-progress",
    )
    assert result.exit_code == 0, result.output
    mask = tmp_path / "mask.zarr"
    result = invoke(
        "mask", "generate", preprocessed_store, thresholds, "-o", mask, "--no-progress"
    )
    assert result.exit_code == 0, result.output
    sets = xr.open_zarr(mask, consolidated=True)["sample_set"].values.tolist()
    assert sets == ["ALL", "pop1", "pop2", "pop3"]


def test_preprocess_thresholds_does_not_overwrite_without_force(
    invoke, preprocessed_store, tmp_path
):
    output = tmp_path / "thresholds.yml"
    output.write_text("keep me\n")
    args = ("preprocess", "thresholds", preprocessed_store, "-o", output)
    result = invoke(*args, "--no-progress")
    assert result.exit_code != 0
    assert "--force" in result.output
    assert output.read_text() == "keep me\n"
    result = invoke(*args, "--force", "--no-progress")
    assert result.exit_code == 0, result.output
    assert "keep me" not in output.read_text()


def test_preprocess_thresholds_without_output_writes_no_file(
    invoke, preprocessed_store, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = invoke("preprocess", "thresholds", preprocessed_store, "--no-progress")
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.yml")) == []


def test_preprocess_thresholds_writes_no_file_without_the_cutoff(
    invoke, preprocessed_store, tmp_path
):
    output = tmp_path / "thresholds.yml"
    result = invoke(
        "preprocess",
        "thresholds",
        preprocessed_store,
        "--threshold",
        9,
        "-o",
        output,
        "--no-progress",
    )
    assert result.exit_code != 0
    assert "cutoff 9" in result.output
    assert not output.exists()


@pytest.mark.parametrize("max_positions", [0, 5])
def test_preprocess_thresholds_accepts_max_positions(
    invoke, preprocessed_store, tmp_path, max_positions
):
    output = tmp_path / "thresholds.yml"
    result = invoke(
        "preprocess",
        "thresholds",
        preprocessed_store,
        "-o",
        output,
        "--max-positions",
        max_positions,
        "--no-progress",
    )
    assert result.exit_code == 0, result.output
    ds = DataStore(preprocessed_store)
    expected = profile_from_defaults(
        ds.tracks["depth"],
        compute_threshold_defaults(ds, "depth", max_positions=max_positions),
        ds.id,
        missing_cutoff=3,
    )
    assert ThresholdProfile.read(output) == expected


def test_max_positions_must_not_be_negative(invoke, preprocessed_store):
    result = invoke(
        "preprocess", "thresholds", preprocessed_store, "--max-positions", -1
    )
    assert result.exit_code == 2
    assert "--max-positions" in result.output


@pytest.mark.parametrize("command", [("serve",), ("preprocess", "thresholds")])
def test_max_positions_is_documented(invoke, command):
    result = invoke(*command, "--help")
    assert result.exit_code == 0
    assert "--max-positions" in result.output
    assert "50,000,000" in result.output


@pytest.mark.parametrize(
    ("args", "show"), [((), False), (("--show",), True), (("--no-show",), False)]
)
def test_serve_opens_a_browser_only_when_asked(
    invoke, preprocessed_store, monkeypatch, args, show
):
    import pbview.app

    served = {}
    monkeypatch.setattr(pbview.app, "serve", lambda *a, **kw: served.update(kw))
    result = invoke("serve", preprocessed_store, *args)
    assert result.exit_code == 0, result.output
    assert served["show"] is show


D4_FILES = sorted((Path(__file__).parent / "data").glob("s[1-7].per-base.d4"))


def depth_chunks(store):
    """Chunk shape (positions, samples) of the imported depth track."""
    meta = json.loads((Path(store) / "depth" / "values" / "zarr.json").read_text())
    return tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])


def test_import_chooses_the_position_chunk_from_the_target_bytes(
    invoke, tmp_path, monkeypatch
):
    assert len(D4_FILES) == 7
    # The test data has a few million positions: keep the chunks big, a
    # chunk of a few positions would write hundreds of thousands of files.
    # 7 samples x 4 bytes x 900,000 positions
    monkeypatch.setattr(config, "TARGET_BYTES", 7 * 4 * 900_000)
    monkeypatch.setattr(config, "MIN_POSITION_CHUNK_SIZE", 1)
    store = tmp_path / "auto"
    result = invoke("import", store, *D4_FILES, "--no-progress")
    assert result.exit_code == 0, result.output
    assert depth_chunks(store) == (900_000, 7)  # all samples in every chunk


def test_import_chunk_size_option_overrides_the_default(invoke, tmp_path):
    store = tmp_path / "explicit"
    result = invoke(
        "import", store, *D4_FILES, "--chunk-size", 1_500_000, "--no-progress"
    )
    assert result.exit_code == 0, result.output
    assert depth_chunks(store) == (1_500_000, 7)


@pytest.mark.parametrize("value", [0, -5, "many"])
def test_import_chunk_size_must_be_a_positive_integer(invoke, tmp_path, value):
    result = invoke("import", tmp_path / "x", *D4_FILES, "--chunk-size", value)
    assert result.exit_code == 2
    assert "--chunk-size" in result.output


def test_import_help_describes_the_chunk_size(invoke):
    result = invoke("import", "--help")
    assert "Every chunk holds all samples" in " ".join(result.output.split())


def test_preprocess_sum_runs(invoke, make_datastore_path, info):
    path = make_datastore_path()
    result = invoke(
        "preprocess", "sum", path, "--sampleset-info", info, "--no-progress"
    )
    assert result.exit_code == 0, result.output
    assert DataStore(path).summaries_of("depth").sum is not None


def read_tsv(path):
    return pd.read_csv(path, sep="\t")


def test_pca_writes_scores_and_variance(invoke, preprocessed_store, tmp_path):
    prefix = tmp_path / "out"
    result = invoke("pca", preprocessed_store, "-o", prefix, "--window-size", 4)
    assert result.exit_code == 0, result.output
    assert "7 samples in 5 windows of 4 bp" in result.output
    scores = read_tsv(f"{prefix}.scores.tsv")
    assert list(scores.columns) == ["sample", "sample_sets"] + [
        f"PC{i}" for i in range(1, 8)
    ]
    assert scores["sample"].tolist() == [f"s{i}" for i in range(1, 8)]
    assert scores["sample_sets"].tolist() == [
        "pop1",
        "pop1",
        "pop1",
        "pop2",
        "pop2",
        "pop3",
        "pop3",
    ]
    variance = read_tsv(f"{prefix}.variance.tsv")
    assert variance["component"].tolist() == [f"PC{i}" for i in range(1, 8)]
    assert variance["explained_variance_ratio"].sum() == pytest.approx(1.0)
    # the same numbers as the library
    ds = DataStore(preprocessed_store)
    track = ds.tracks["depth"]
    expected = track.pca(Coordinates().resolve_for(track, ds.genome), 4)
    np.testing.assert_allclose(scores["PC1"], expected.loadings[:, 0], rtol=1e-6)


def test_pca_selection_options(invoke, preprocessed_store, tmp_path):
    prefix = tmp_path / "subset"
    result = invoke(
        "pca",
        preprocessed_store,
        "-o",
        prefix,
        "--window-size",
        3,
        "--sample-set",
        "pop1",
        "--sample-set",
        "pop2",
        "--min-contig-length",
        8,
        "--n-components",
        2,
    )
    assert result.exit_code == 0, result.output
    scores = read_tsv(f"{prefix}.scores.tsv")
    assert scores["sample"].tolist() == ["s1", "s2", "s3", "s4", "s5"]
    assert list(scores.columns)[-2:] == ["PC1", "PC2"]
    assert len(read_tsv(f"{prefix}.variance.tsv")) == 2
    # chr10 (6 bp) is left out: chr1 gives 3 windows and chr2 gives 2
    assert "in 5 windows of 3 bp" in result.output


def test_pca_does_not_overwrite_without_force(invoke, preprocessed_store, tmp_path):
    prefix = tmp_path / "out"
    args = ("pca", preprocessed_store, "-o", prefix, "--window-size", 4)
    assert invoke(*args).exit_code == 0
    again = invoke(*args)
    assert again.exit_code != 0
    assert "--force" in again.output
    assert "Traceback" not in again.output
    assert invoke(*args, "--force").exit_code == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--window-size", 11), "at least two windows"),
        (("--window-size", 0), "--window-size"),
        (("--window-size", 4, "--sample-set", "nope"), "nope"),
        (("--window-size", 4, "--track-name", "missing"), "no track 'missing'"),
        (("--window-size", 4, "--n-components", 0), "--n-components"),
    ],
)
def test_pca_reports_problems_cleanly(
    invoke, preprocessed_store, tmp_path, args, message
):
    result = invoke("pca", preprocessed_store, "-o", tmp_path / "x", *args)
    assert "Traceback" not in result.output
    assert result.exit_code != 0
    assert message in result.output
