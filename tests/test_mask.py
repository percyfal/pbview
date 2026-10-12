import io

import pytest
import xarray as xr

import pbview.mask.export as _export


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
