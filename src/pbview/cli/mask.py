"""Accessible sites creation for pbzarr stores."""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-23"

from typing import Callable, TextIO  # noqa

from pathlib import Path
import xarray as xr

import click
from click.decorators import FC

from pbview.logging import log_level
from pbview.mask import generate as _generate
from pbview.mask import export as _export

from ._common import (
    memory_limit_option,
    path_argument,
    progress_option,
    sampleinfo_option,
    sampleset_name_option,
    set_threads,
    threads_option,
    track_name_option,
    use_dask_option,
    workers_option,
)


def mask_output_format_option(default: str = "BED") -> Callable[[FC], FC]:
    return click.option(
        "-O",
        "--output-format",
        default=default,
        type=click.Choice(["BED", "FASTA"], case_sensitive=False),
        help="output file format",
    )


@click.group()
def mask() -> None:
    """Generate and summarize masks."""
    pass


@click.command()
@path_argument(exists=True, dir_okay=True, nargs=1)
@click.argument(
    "threshold",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1,
    required=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=True),
    help="Output zarr store",
)
@threads_option(default=1)
@workers_option(default=1)
@memory_limit_option()
@track_name_option()
@log_level()
@progress_option()
@sampleinfo_option()
@use_dask_option()
@set_threads
def generate(
    path: Path | str,
    threshold: Path | str,
    output: Path | str,
    track_name: str,
    progress: bool,
    sampleinfo: str | None = None,
) -> None:
    """Generate accessible sites from the pbzarr store.

    Generate accessible sites mask and exclusion masks for the pbzarr
    store at PATH based on threshold values in THRESHOLD. Generated
    masks are output in a separate Zarr archive, defaulting to
    PATH_mask.

    """
    output = output or Path(path).with_name(f"{Path(path).name}_mask")
    _generate.generate_masks(
        path,
        sampleinfo=sampleinfo,
        threshold=threshold,
        output=output,
        track_name=track_name,
        progress=progress,
    )


@click.command()
@path_argument(exists=True, dir_okay=True, nargs=1)
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default="-",
    help="Output file name",
)
@mask_output_format_option()
@threads_option(default=1)
@workers_option(default=1)
@memory_limit_option()
@track_name_option(default="combined")
@sampleset_name_option()
@log_level()
@progress_option()
@use_dask_option()
@set_threads
def export(
    path: Path | str,
    output: TextIO,
    output_format: str,
    track_name: str,
    progress: bool,
    sampleset_name: str | None = None,
) -> None:
    """Export mask to BED or FASTA format.

    Export mask to BED or FASTA format. Accesible sites are translated
    to regions in BED export, and to 0/1 sequence masks in FASTA
    export where 1 indicates inclusion, 0 exclusion. The default
    output file name is based on the input path, defaulting to
    PATH_MASK.bed or PATH_MASK.fasta.

    """
    ds = xr.open_zarr(path, consolidated=True)

    _export.export_mask(
        ds,
        sampleset_name=sampleset_name,
        output=output,
        output_format=output_format,
        track_name=track_name,
        progress=progress,
    )


mask.add_command(generate)
mask.add_command(export)
