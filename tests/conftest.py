import os
from pathlib import Path

import pandas as pd
import panel as pn
import pytest
from pytest import fixture

from pbview.importers import d4 as import_d4
from pbview.preprocess import preprocess

dirname = Path(os.path.abspath(os.path.dirname(__file__)))

PORT = [6000]


def pytest_configure(config):
    pytest.dname = dirname


@fixture
def port():
    PORT[0] += 1
    return PORT[0]


@fixture(autouse=True)
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
def store(tmpdir_factory, d4all, sampleinfo):
    p = Path(tmpdir_factory.mktemp("store")) / "datastore.zarr"
    d4all = [str(p) for p in d4all]
    import_d4.ingest(str(p), d4all)
    preprocess.preprocess(p, sampleinfo=sampleinfo)
    return p


@pytest.fixture(scope="session")
def sampleinfo():
    return pytest.dname / "data" / "sampleinfo.tsv"


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
