from pathlib import Path

import click

from pbview.importers import d4
from pbview.logging import cli_logger as logger
from pbview.logging import log_level

from ._common import (
    chunk_size_option,
    d4_argument,
    path_argument,
    progress_option,
    track_name_option,
    workers_option,
)


@click.command("import")
@path_argument(exists=False, dir_okay=True, nargs=1)
@d4_argument(nargs=-1)
@workers_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
def import_d4(
    path: Path | str,
    d4files: list[Path] | list[str],
    track_name: str,
    workers: int,
    progress: bool,
    chunk_size: int,
) -> None:
    """Import d4 files into a pbzarr store."""
    if len(d4files) == 0:
        logger.error("Provide at least one d4 source file to import")
        return
    d4.ingest(
        path,
        d4=d4files,
        track=track_name,
        workers=workers,
        progress=progress,
        chunk_size=chunk_size,
    )
