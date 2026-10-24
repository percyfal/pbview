import os
from pathlib import Path

import _helpers
import pandas as pd
import panel as pn
import pytest
import xarray as xr

from pbview.importers import d4 as import_d4
from pbview.model.datastore import DataStore
from pbview.preprocess import preprocess

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
            default = tree[f"{track_name}/_meta/column_grouping"]["values"]
            ss = xr.Dataset(
                {"values": xr.concat([default, sample_sets], dim="sample_set")}
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
            default = tree[f"{track_name}/_meta/column_grouping"]["values"]
            ss = xr.Dataset(
                {"values": xr.concat([default, sample_sets], dim="sample_set")}
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


def pytest_configure(config):
    pytest.dname = dirname


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


@pytest.fixture(scope="session")
def d4file():
    def _d4file(name):
        return pytest.dname / "data" / f"{name}.per-base.d4"

    return _d4file


@pytest.fixture(scope="session")
def d4all(d4file) -> list[Path]:
    return [d4file(f"s{i}") for i in range(1, 8)]


@pytest.fixture(scope="session")
def store(tmpdir_factory, d4all, sampleset_info):
    p = Path(tmpdir_factory.mktemp("store")) / "datastore.zarr"
    d4all = [str(p) for p in d4all]
    import_d4.ingest(str(p), d4all)
    preprocess.preprocess(p, sampleset_info=sampleset_info)
    return p


@pytest.fixture(scope="session")
def sampleset_info():
    return pytest.dname / "data" / "sampleset_info.tsv"


@pytest.fixture(scope="session")
def annotation():
    return pytest.dname / "data" / "annotation.gff.gz"


@pytest.fixture(scope="session")
def annotation_df():
    return pd.DataFrame(
        {
            "seqid": ["chr1", "chr1", "chr2", "chr2"],
            "source": ["d4explorer"] * 4,
            "type": ["gene", "rRNA", "gene", "exon"],
            "start": [10, 10, 20, 20],
            "end": [100, 100, 60, 60],
            "score": ["."] * 4,
            "strand": ["+", "+", "-", "-"],
            "phase": ["."] * 4,
            "attributes": [
                "ID=chr1_G000001;product=5S ribosomal RNA",
                ("ID=chr1_G000001.rRNA.1;Parent=chr1_G000001;product=5S ribosomal RNA"),
                "ID=chr2_G000001;product=exon 1",
                "ID=chr2_G000001.exon.1;Parent=chr2_G000001;product=exon 1",
            ],
        }
    )


@pytest.fixture(scope="session")
def annotation_df_path(annotation_df, tmpdir_factory):
    p = tmpdir_factory.mktemp("annotation")
    outfile = p / "annotation.gff"
    annotation_df.to_csv(str(outfile), index=False, header=None, sep="\t")
    return Path(outfile)
