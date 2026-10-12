import textwrap
from contextlib import nullcontext
from typing import TextIO

import click
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from pbview.logging import cli_logger as logger


def _export_mask_bed(data, contigs, offsets, output: TextIO):
    for chrom, start, end in zip(contigs, offsets[0:-1], offsets[1:]):
        selected = data[start:end].astype(bool)
        if not np.any(selected):
            continue
        diff = np.diff(np.concatenate(([0], selected, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            output.write(f"{chrom}\t{s}\t{e}\n")


def _export_mask_fasta(data, contigs, offsets, output: TextIO):
    for chrom, start, end in zip(contigs, offsets[0:-1], offsets[1:]):
        selected = data[start:end].astype(bool)
        output.write(f">{chrom}\n")
        seq = textwrap.wrap("".join(np.int8(selected).astype(str)), width=70)
        output.write("\n".join(seq))
        output.write("\n")


def export_mask(
    ds: xr.Dataset,
    *,
    sampleset_name: str | None = None,
    output_format: str | None = "BED",
    output: TextIO | None = None,
    track_name: str | None = None,
    progress: bool = False,
):
    try:
        track = ds[track_name]
    except KeyError:
        raise click.BadParameter(f"Track '{track_name}' not found in datastore.")
    try:
        data = track.sel(sample_set=sampleset_name).values
    except KeyError:
        raise click.BadParameter(
            f"Sample set '{sampleset_name}' not found in track '{track_name}'."
        )

    logger.info(
        "Exporting mask for track '%s', sample set '%s' in format '%s'",
        track_name,
        sampleset_name,
        output_format,
    )
    with ProgressBar() if progress else nullcontext():
        if output_format.upper() == "BED":
            _export_mask_bed(
                data, ds.coords["contigs"].values, ds.coords["offsets"].values, output
            )
        elif output_format.upper() == "FASTA":
            _export_mask_fasta(
                data, ds.coords["contigs"].values, ds.coords["offsets"].values, output
            )
