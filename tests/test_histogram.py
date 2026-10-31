import numpy as np
import pytest

from pbview.model.coordinates import Coordinates
from pbview.model.histogram import compute_threshold_defaults, fold_hist


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
