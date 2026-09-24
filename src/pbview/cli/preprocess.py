from pathlib import Path

import click

from pbview.logging import log_level
from pbview.preprocess import preprocess as preprocess_mod

from ._common import (
    chunk_size_option,
    path_argument,
    progress_option,
    sampleinfo_option,
    set_threads,
    threads_option,
    track_name_option,
    workers_option,
)


@click.command()
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@threads_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
@sampleinfo_option()
@set_threads
def preprocess(
    path: Path | str,
    track_name: str,
    progress: bool,
    chunk_size: int,
    sampleinfo: str | None = None,
) -> None:
    """Preprocess the pbzarr store for faster access."""
    preprocess_mod.preprocess(
        path,
        track=track_name,
        progress=progress,
        chunk_size=chunk_size,
        sampleinfo=sampleinfo,
    )
