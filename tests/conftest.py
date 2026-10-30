import os
from pathlib import Path

import _helpers
import panel as pn
import pytest
import xarray as xr

from pbview.model.datastore import DataStore

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
