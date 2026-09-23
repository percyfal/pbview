import re
from pathlib import Path

import click
import pbzarr

from pbview.logging import app_logger as logger
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
    d4: list[Path] | list[str],
    track_name: str,
    workers: int,
    progress: bool,
    chunk_size: int,
) -> None:
    """Import d4 files into a pbzarr store."""
    if len(d4) == 0:
        logger.error("Provide at least one d4 source file to import")
        return
    run_import_d4(
        path,
        d4=d4,
        track=track_name,
        workers=workers,
        progress=progress,
        chunk_size=chunk_size,
    )


def run_import_d4(
    path: Path | str,
    d4: list[Path] | list[str],
    *,
    track: str = "depth",
    workers: int = 1,
    progress: bool = True,
    chunk_size: int = 1_000_000,
) -> None:
    """Import d4 files to pbzarr store.

    This function will create a new pbzarr store at `path` and add
    tracks by importing data from the specified sources.
    """
    # Create the pbzarr store
    if Path(path).exists():
        logger.debug("Path %s already exists, not creating store", path)
    else:
        logger.info("Creating pbzarr store at %s", path)
        pbzarr.create_store(path)

    sources: list[tuple[str, str]] = []
    print("Sources: ", sources)
    for fn in d4:
        if not re.search(r".d4$", fn):
            raise ValueError(f"Input file {fn} does not have .d4 extension")
        fn = Path(fn)
        sample = re.sub(".per-base.d4", "", fn.name)
        sources.append((str(fn), sample))

    # Import data into the store
    logger.info("Importing %i data sources", len(sources))
    n_samples = len(sources)
    try:
        report = pbzarr.import_d4(
            destination=str(path),
            track=track,
            sources=sources,
            workers=workers,
            progress=progress,
            chunk_size=chunk_size,
            column_dim="sample",
            column_chunk_size=n_samples,
        )
    except pbzarr.PbzError as e:
        logger.error("Error importing d4 files: %s", e)
        raise
    except Exception as e:
        logger.error("Error importing d4 files: %s", e)
        logger.error(report)
        raise
