import io

import pytest
import xarray as xr

import pbview.mask.export as _export
from pbview import config
from pbview.mask.generate import generate_mask
from pbview.model.thresholds import SampleSetThresholds, ThresholdProfile


@pytest.fixture()
def tiny_dataset() -> xr.Dataset:
    ds = xr.Dataset(
        coords={
            "sample_set": ["set1", "set2"],
            "contigs": ["chr1", "chr2"],
            "offsets": [0, 6, 10],
        },
        data_vars={
            "combined": (
                ("sample_set", "position"),
                [[1, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1, 1]],
            ),
        },
    )
    ds["combined"] = ds["combined"].astype("bool")
    return ds


@pytest.mark.parametrize(
    "sample_set,expected_lines",
    [
        ("set1", ["chr1\t0\t1", "chr1\t2\t4", "chr2\t0\t2"]),
        ("set2", ["chr1\t1\t3", "chr1\t5\t6", "chr2\t0\t1", "chr2\t2\t4"]),
    ],
)
def test_export_mask_bed(tiny_dataset, sample_set, expected_lines):
    ds = tiny_dataset
    buf = io.StringIO()
    data = ds["combined"].sel(sample_set=sample_set).values
    _export._export_mask_bed(
        data, ds.coords["contigs"].values, ds.coords["offsets"].values, buf
    )
    bed = buf.getvalue().strip().splitlines()
    assert len(bed) == len(expected_lines)
    for line, expected in zip(bed, expected_lines):
        assert line == expected


@pytest.mark.parametrize(
    "sample_set,expected_lines",
    [
        ("set1", [">chr1", "101100", ">chr2", "1100"]),
        ("set2", [">chr1", "011001", ">chr2", "1011"]),
    ],
)
def test_export_mask_fasta(tiny_dataset, sample_set, expected_lines):
    ds = tiny_dataset
    buf = io.StringIO()
    data = ds["combined"].sel(sample_set=sample_set).values
    _export._export_mask_fasta(
        data, ds.coords["contigs"].values, ds.coords["offsets"].values, buf
    )
    bed = buf.getvalue().strip().splitlines()
    assert len(bed) == len(expected_lines)
    for line, expected in zip(bed, expected_lines):
        assert line == expected


def test_generate_mask(make_datastore_path, sample_sets, tmp_path):
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    sets = ["ALL", "pop1", "pop2", "pop3"]
    profile = ThresholdProfile(
        schema_version=config.SCHEMA_VERSION,
        dataset_id="test",
        sample_sets={s: SampleSetThresholds(3, 10, 200, 1) for s in sets},
    )
    thresholds = tmp_path / "thresholds.yml"
    profile.write(thresholds)
    output = tmp_path / "mask.zarr"
    generate_mask(path, threshold=thresholds, output=output, progress=False)
    mask = xr.open_zarr(output, consolidated=True)
    assert mask["sample_set"].values.tolist() == sets
    assert mask["accessible"].dims == ("position", "sample_set")
    assert mask.sizes["position"] == 24
