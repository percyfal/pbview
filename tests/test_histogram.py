import numpy as np
import pytest

from pbview.model.datastore import DataStore
from pbview.model.histogram import compute_threshold_defaults


@pytest.fixture()
def ds(store, sampleinfo):
    return DataStore(path=store, sampleinfo=sampleinfo)


@pytest.fixture()
def track(ds, store):
    return ds.tracks["depth"]


def test_compute_threshold_defaults(ds):
    defaults = compute_threshold_defaults(ds, "depth")
    assert defaults == {
        'lower_coverage_ALL': 61,
        'upper_coverage_ALL': 226,
        'maxbins_ALL': 319,
        'lower_coverage_pop1': 24,
        'upper_coverage_pop1': 93,
        'maxbins_pop1': 142,
        'lower_coverage_pop2': 21,
        'upper_coverage_pop2': 80,
        'maxbins_pop2': 112,
        'lower_coverage_pop3': 16,
        'upper_coverage_pop3': 57,
        'maxbins_pop3': 90
    }


