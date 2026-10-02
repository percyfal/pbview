"""
Create annotation mask for a pbzarr store.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-24"

from pathlib import Path

import click

from pbview.logging import log_level

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
def annotation() -> None:
    """Import and generate annotation tracks for a pbzarr store."""
    pass


# FIXME: add options for bed import, name of annotation, etc
@click.command("import")
@path_argument(exists=False, dir_okay=True, nargs=1)
@threads_option(default=1)
@workers_option(default=1)
@memory_limit_option()
@track_name_option()
@log_level()
@progress_option()
@sampleinfo_option()
@use_dask_option()
@set_threads
def import_annotation(
    path: Path | str,
    output: Path | str,
    track_name: str,
    progress: bool,
    sampleinfo: str | None = None,
) -> None:
    """Import annotation tracks to PATH."""
    output = output or Path(path).with_name(f"{Path(path).name}_mask")


annotation.add_command(import_annotation)
