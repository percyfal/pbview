"""
Mask creation for pbzarr stores.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-23"

from pathlib import Path

import click

from pbview.logging import log_level
from pbview.mask import generate as _generate

from ._common import (
    memory_limit_option,
    path_argument,
    progress_option,
    sampleinfo_option,
    set_threads,
    threads_option,
    track_name_option,
    use_dask_option,
    workers_option,
)


@click.group()
def mask() -> None:
    """Generate and summarize masks"""
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
    """Generate masks for the pbzarr store.

    Generate masks for the pbzarr store at PATH based on threshold
    values in THRESHOLD.
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


mask.add_command(generate)
