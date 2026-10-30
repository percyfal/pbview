from pbview.model.histogram import compute_threshold_defaults


# FIXME: compute_threshold_defaults returns defaults for ALL sampleset
# only, not per sampleset
def test_compute_threshold_defaults(full_ds):
    defaults = compute_threshold_defaults(full_ds, "depth")
    assert defaults == {
        "lower_coverage_ALL": 73,
        "upper_coverage_ALL": 229,
        "maxbins_ALL": 183,
    }
