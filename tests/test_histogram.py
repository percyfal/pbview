import dask
import numpy as np
import pytest

from pbview import config
from pbview.model.coordinates import Coordinates
from pbview.model.histogram import (
    compute_threshold_defaults,
    fold_hist,
    hist_quantile,
    hist_stats,
    sample_set_hists,
)


def test_compute_threshold_defaults(full_ds):
    defaults = compute_threshold_defaults(full_ds, "depth")
    assert defaults == {
        "lower_coverage_ALL": 73,
        "upper_coverage_ALL": 229,
        "maxbins_ALL": 183,
        "lower_coverage_pop1": 27,
        "upper_coverage_pop1": 90,
        "maxbins_pop1": 80,
        "lower_coverage_pop2": 26,
        "upper_coverage_pop2": 83,
        "maxbins_pop2": 67,
        "lower_coverage_pop3": 19,
        "upper_coverage_pop3": 59,
        "maxbins_pop3": 60,
    }


@pytest.mark.parametrize("n_bins", [1, 3, 10, 40, 400])
def test_fold_hist_equals_rebinning_clipped_values(n_bins):
    values = np.random.default_rng(0).integers(0, 60, size=500)
    expected = np.bincount(np.minimum(values, n_bins - 1), minlength=n_bins)
    assert np.array_equal(fold_hist(np.bincount(values), n_bins), expected)


def test_defaults_derive_the_clipped_histogram_from_one_pass(full_ds):
    """`upper` used to come from a second pass, `coverage_hist(arange(max_bin))`."""
    track = full_ds.tracks["depth"]
    defaults = compute_threshold_defaults(full_ds, "depth")
    for s in track.groups("sample_set"):
        resolved = (
            Coordinates().with_sample_sets([s]).resolve_for(track, full_ds.genome)
        )
        mb = defaults[f"maxbins_{s}"]
        second_pass, _ = track.coverage_hist(bins=np.arange(mb), resolved=resolved)
        h = track.sum_hist(s, len(resolved.labels))
        assert np.array_equal(fold_hist(h, mb), second_pass)


def _expected_defaults(sums, max_bin_q=0.995, headroom=1.2):
    """lower, upper, max_bin for one sample set, from numpy sums"""
    h = np.bincount(sums)
    lower = int(0.6 * hist_stats(h)["mean"])
    max_bin = int(hist_quantile(h, max_bin_q) * headroom)
    clipped = np.bincount(np.minimum(sums, max_bin - 1), minlength=max_bin)
    st = hist_stats(clipped)
    return lower, int(st["median"] + 2 * st["std"]), max_bin


def test_defaults_are_estimated_from_every_kth_position(full_ds):
    sums = full_ds.tracks["depth"].summaries.sum.column("ALL").values  # 24
    for max_positions, stride in ((5, 5), (23, 2), (24, 1), (0, 1)):
        d = compute_threshold_defaults(full_ds, "depth", max_positions=max_positions)
        lower, upper, max_bin = _expected_defaults(sums[::stride])
        assert (
            d["lower_coverage_ALL"],
            d["upper_coverage_ALL"],
            d["maxbins_ALL"],
        ) == (lower, upper, max_bin), max_positions


def test_defaults_max_positions_comes_from_the_config(full_ds, monkeypatch):
    exact = compute_threshold_defaults(full_ds, "depth", max_positions=0)
    assert compute_threshold_defaults(full_ds, "depth") == exact  # 24 < 50 million
    monkeypatch.setattr(config, "MAX_POSITIONS", 5)
    estimate = compute_threshold_defaults(full_ds, "depth")
    assert estimate == compute_threshold_defaults(full_ds, "depth", max_positions=5)
    assert estimate != exact


def test_threshold_defaults_and_the_table_share_one_read(full_ds, monkeypatch):
    calls = []
    real = dask.compute
    monkeypatch.setattr(
        dask, "compute", lambda *a, **k: calls.append(len(a)) or real(*a, **k)
    )
    compute_threshold_defaults(full_ds, "depth", max_positions=5)
    hists = sample_set_hists(full_ds, "depth", 5)  # as compute_thresholds does
    assert calls == [4]  # four sample sets, one pass
    assert list(hists) == ["ALL", "pop1", "pop2", "pop3"]
