import io

import _helpers
import click
import numpy as np
import pytest
import xarray as xr

import pbview.mask.export as _export
from pbview import config
from pbview.mask.generate import generate_mask
from pbview.model.thresholds import SampleSetThresholds, ThresholdProfile
from pbview.model.track import MissingSummaryError


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


@pytest.fixture
def several_cutoff_mask(two_cutoff_store, threshold_file, tmp_path):
    output = tmp_path / "mask.zarr"
    generate_mask(
        two_cutoff_store,
        threshold=threshold_file,
        output=output,
        progress=False,
        missing_cutoffs=[5, 3, 5],
    )
    return xr.open_zarr(output, consolidated=True)


def test_generate_mask_for_several_cutoffs(several_cutoff_mask):
    mask = several_cutoff_mask
    assert mask["missing_cutoff"].values.tolist() == [3, 5]
    assert mask.attrs["missing_cutoffs"] == [3, 5]
    for layer in ("accessible", "excluded_missing"):
        assert mask[layer].dims == ("position", "sample_set", "missing_cutoff")
    assert mask["excluded_coverage"].dims == ("position", "sample_set")
    for cutoff in (3, 5):
        for name, expected in _helpers.expected_accessible(cutoff).items():
            got = mask["accessible"].sel(sample_set=name, missing_cutoff=cutoff)
            np.testing.assert_array_equal(got.values, expected)


def test_a_larger_cutoff_never_adds_accessible_sites(several_cutoff_mask):
    accessible = several_cutoff_mask["accessible"]
    assert not (
        accessible.sel(missing_cutoff=5) & ~accessible.sel(missing_cutoff=3)
    ).any()
    assert (accessible.sel(missing_cutoff=3) != accessible.sel(missing_cutoff=5)).any()


def test_generate_mask_without_cutoffs_keeps_the_threshold_file_cutoff(
    two_cutoff_store, threshold_file, tmp_path
):
    output = tmp_path / "plain.zarr"
    generate_mask(
        two_cutoff_store, threshold=threshold_file, output=output, progress=False
    )
    mask = xr.open_zarr(output, consolidated=True)
    assert "missing_cutoff" not in mask.dims
    for name, expected in _helpers.expected_accessible(3).items():
        got = mask["accessible"].sel(sample_set=name)
        np.testing.assert_array_equal(got.values, expected)


def test_generate_mask_requires_the_missingness_summary(
    two_cutoff_store, threshold_file, tmp_path
):
    with pytest.raises(MissingSummaryError, match="cutoff 7"):
        generate_mask(
            two_cutoff_store,
            threshold=threshold_file,
            output=tmp_path / "x.zarr",
            progress=False,
            missing_cutoffs=[7],
        )


def _bed(mask, **kwargs):
    buf = io.StringIO()
    _export.export_mask(
        mask, output=buf, output_format="BED", track_name="accessible", **kwargs
    )
    return buf.getvalue()


def _expected_bed(mask, cutoff, sample_set="ALL"):
    buf = io.StringIO()
    _export._export_mask_bed(
        _helpers.expected_accessible(cutoff)[sample_set],
        mask.coords["contigs"].values,
        mask.coords["offsets"].values,
        buf,
    )
    return buf.getvalue()


def test_export_picks_the_requested_cutoff(several_cutoff_mask):
    mask = several_cutoff_mask
    exported = {c: _bed(mask, sampleset_name="ALL", missing_cutoff=c) for c in (3, 5)}
    for cutoff, bed in exported.items():
        assert bed == _expected_bed(mask, cutoff)
    assert exported[3] != exported[5]


def test_export_needs_a_cutoff_for_a_several_cutoff_mask(several_cutoff_mask):
    with pytest.raises(click.BadParameter, match="several missing-data thresholds"):
        _bed(several_cutoff_mask, sampleset_name="ALL")


def test_export_rejects_an_unknown_cutoff(several_cutoff_mask):
    with pytest.raises(click.BadParameter, match=r"not found .*available: \[3, 5\]"):
        _bed(several_cutoff_mask, sampleset_name="ALL", missing_cutoff=9)


def test_export_cutoff_does_not_apply_to_the_coverage_layer(several_cutoff_mask):
    with pytest.raises(click.BadParameter, match="no missing-data threshold"):
        _export.export_mask(
            several_cutoff_mask,
            sampleset_name="ALL",
            missing_cutoff=3,
            output=io.StringIO(),
            track_name="excluded_coverage",
        )


def test_export_cutoff_needs_a_several_cutoff_mask(
    two_cutoff_store, threshold_file, tmp_path
):
    output = tmp_path / "plain.zarr"
    generate_mask(
        two_cutoff_store, threshold=threshold_file, output=output, progress=False
    )
    mask = xr.open_zarr(output, consolidated=True)
    with pytest.raises(click.BadParameter, match=r"threshold file's cutoff \[3\]"):
        _bed(mask, sampleset_name="ALL", missing_cutoff=3)
