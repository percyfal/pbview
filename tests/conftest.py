import os
from pathlib import Path

import _helpers
import panel as pn
import pytest
import xarray as xr

from pbview import config
from pbview.model.datastore import DataStore
from pbview.model.thresholds import SampleSetThresholds, ThresholdProfile
from pbview.preprocess import preprocess as preprocess_mod

dirname = Path(os.path.abspath(os.path.dirname(__file__)))

PORT = [6000]


@pytest.fixture
def tiny_datatree() -> xr.DataTree:
    return _helpers.tiny_datatree()


@pytest.fixture
def sample_sets() -> xr.DataArray:
    return _helpers.sample_sets()


@pytest.fixture
def make_datastore_path(tmp_path, tiny_datatree):
    """Build datastore path under different settings"""

    def _make(
        with_sum: bool = False,
        with_missing: bool = False,
        sample_sets: xr.DataArray | None = None,
        track_name: str = "depth",
    ):
        state = ["raw"] if (not with_sum and not with_missing) else ["preprocessed"]
        state += ["with_sum"] if with_sum else []
        state += ["with_missing"] if with_missing else []
        state += ["with_sample_sets"] if sample_sets is not None else []
        path = tmp_path / f"store_{'_'.join(state)}.zarr"
        return _helpers.make_store(
            path,
            sample_sets=sample_sets,
            with_sum=with_sum,
            with_missing=with_missing,
            track_name=track_name,
        )

    return _make


@pytest.fixture
def make_datastore(make_datastore_path):
    def _make(**kw):
        path = make_datastore_path(**kw)
        ds = DataStore(path)
        return ds

    return _make


@pytest.fixture
def datastore(make_datastore_path):
    path = make_datastore_path()
    return DataStore(path)


@pytest.fixture
def raw_ds(make_datastore_path):
    path = make_datastore_path()
    return DataStore(path)


@pytest.fixture
def full_ds(make_datastore_path, sample_sets):
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    return DataStore(path)


@pytest.fixture
def port():
    PORT[0] += 1
    return PORT[0]


@pytest.fixture(autouse=True)
def server_cleanup():
    """Cleanup server after test."""
    try:
        yield
    finally:
        pn.state.reset()


@pytest.fixture
def two_cutoff_store(make_datastore_path, sample_sets):
    """Store with sample sets, a sum and missingness summaries for cutoffs 3 and 5"""
    path = make_datastore_path(sample_sets=sample_sets)
    preprocess_mod.preprocess(path, progress=False, missing_cutoffs=(3, 5))
    return path


@pytest.fixture
def threshold_file(tmp_path):
    """Threshold profile: cutoff 3, coverage 10-200, no missing sample allowed"""
    sets = ["ALL", "pop1", "pop2", "pop3"]
    profile = ThresholdProfile(
        schema_version=config.SCHEMA_VERSION,
        dataset_id="test",
        sample_sets={s: SampleSetThresholds(3, 10, 200, 0) for s in sets},
    )
    path = tmp_path / "thresholds.yml"
    profile.write(path)
    return path
