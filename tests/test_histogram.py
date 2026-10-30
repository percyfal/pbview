from pbview.model.histogram import compute_threshold_defaults


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
