"""
Mask creation for pbzarr stores.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-23"

from pathlib import Path

import click

from pbview.logging import app_logger as logger  # noqa
from pbview.logging import log_level
from pbview.model.datastore import DataStore

from ._common import (
    path_argument,
    progress_option,
    sampleinfo_option,
    track_name_option,
    workers_option,
)


@click.group()
def mask():
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
@workers_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@sampleinfo_option()
def generate(
    path: Path | str,
    threshold: Path | str,
    output: Path | str,
    track_name: str,
    workers: int,
    progress: bool,
    sampleinfo: str | None = None,
) -> None:
    """Generate a mask for the pbzarr store."""
    generate(
        path,
        threshold,
        output=output,
        track=track_name,
        workers=workers,
        progress=progress,
        sampleinfo=sampleinfo,
    )


mask.add_command(generate)


def run_create_mask(
    path: Path | str,
    threshold: Path | str,
    output: Path | str | None = None,
    track: str = "depth",
    progress: bool = True,
    sampleinfo: Path | str | None = None,
) -> None:
    """Create masks for the pbzarr store.

    Create masks for the pbzarr store at PATH based on threshold
    values in THRESHOLD.
    """
    output = output or Path(path).with_name(f"{Path(path).name}_mask")
    ds = DataStore(path=path, sampleinfo=sampleinfo)
    logger.info("Creating mask for track %s based on threshold %s", track, threshold)
    logger.info("Writing mask to pbzarr store at %s", output)
    print(ds)
    # with ProgressBar():
    #     ds.create_mask(
    #         track=ds.tracks[track],
    #         threshold=threshold,
    #         workers=workers,
    #         progress=progress,
    #     )
