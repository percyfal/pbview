from pbview.model.histogram import compute_threshold_defaults


# FIXME: compute_threshold_defaults returns defaults for ALL sampleset
# only, not per sampleset
def test_compute_threshold_defaults(full_ds):
    defaults = compute_threshold_defaults(full_ds, "depth")
    assert defaults == {
        "lower_coverage_ALL": 34,
        "upper_coverage_ALL": 141,
        "maxbins_ALL": 183,
    }
    # {
    #     "lower_coverage_pop1": 24,
    #     "upper_coverage_pop1": 93,
    #     "maxbins_pop1": 142,
    #     "lower_coverage_pop2": 21,
    #     "upper_coverage_pop2": 80,
    #     "maxbins_pop2": 112,
    #     "lower_coverage_pop3": 16,
    #     "upper_coverage_pop3": 57,
    #     "maxbins_pop3": 90,
    # }
