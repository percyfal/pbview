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
def make_datatree(tiny_datatree):
    """Build datatree under different settings"""

    def _make(
        with_default_sample_set: bool = False,
        sample_sets: xr.DataArray | None = None,
        track_name: str = "depth",
    ):
        if with_default_sample_set and sample_sets is not None:
            raise ValueError(
                "Default sample set is implicit when sample_sets is provided."
            )
        tree = tiny_datatree.copy(deep=True)
        if with_default_sample_set:
            tree = _helpers.attach_default_sample_set(tree)
        if sample_sets is not None:
            tree = _helpers.attach_default_sample_set(tree)
            default = tree[f"{track_name}/_meta/column_grouping"]["membership"]
            ss = xr.Dataset(
                {"membership": xr.concat([default, sample_sets], dim="sample_set")}
            )
            ss.attrs["perbase:grouping_hash"] = _helpers.grouping_hash(sample_sets)
            tree[f"{track_name}/_meta/column_grouping"] = ss
        return tree

    return _make


@pytest.fixture
def make_datastore_path(tmp_path, tiny_datatree):
    """Build datastore path under different settings"""

    def _make(
        with_sum: bool = False,
        with_missing: bool = False,
        with_default_sample_set: bool = False,
        with_all: bool = False,
        sample_sets: xr.DataArray | None = None,
        track_name: str = "depth",
    ):
        if with_default_sample_set and sample_sets is not None:
            raise ValueError(
                "Default sample set is implicit when sample_sets is provided."
            )
        if with_all:
            with_sum = with_missing = with_default_sample_set = True
        state = ["raw"] if (not with_sum and not with_missing) else ["preprocessed"]
        state += ["with_sum"] if with_sum else []
        state += ["with_missing"] if with_missing else []
        state += ["with_sample_sets"] if sample_sets is not None else []
        state += ["with_default_sample_set"] if with_default_sample_set else []
        label = "_".join(state)

        path = tmp_path / f"store_{label}.zarr"
        tree = tiny_datatree.copy(deep=True)

        if with_default_sample_set:
            tree = _helpers.attach_default_sample_set(tree)
        if sample_sets is not None:
            tree = _helpers.attach_default_sample_set(tree)
            default = tree[f"{track_name}/_meta/column_grouping"]["membership"]
            ss = xr.Dataset(
                {"membership": xr.concat([default, sample_sets], dim="sample_set")}
            )
            ss.attrs["perbase:grouping_hash"] = _helpers.grouping_hash(sample_sets)
            tree[f"{track_name}/_meta/column_grouping"] = ss
        if with_sum:
            if not with_default_sample_set and sample_sets is None:
                tree = _helpers.attach_default_sample_set(tree)
            tree = _helpers.attach_sum(tree)
        if with_missing:
            if not with_default_sample_set and sample_sets is None:
                _helpers.attach_default_sample_set(tree)
            _helpers.attach_missingness(tree)
        tree.to_zarr(path)
        return path

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
