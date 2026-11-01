import re
from pathlib import Path

import pbzarr

from pbview import config
from pbview.logging import cli_logger as logger
from pbview.model.chunking import position_chunk_size


def ingest(
    path: Path | str,
    d4: list[Path] | list[str],
    *,
    track: str = "depth",
    workers: int = 1,
    progress: bool = True,
    chunk_size: int | None = None,
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
    for fn in d4:
        if not re.search(r".d4$", fn):
            raise ValueError(f"Input file {fn} does not have .d4 extension")
        fn = Path(fn)
        sample = re.sub(".per-base.d4", "", fn.name)
        sources.append((str(fn), sample))
    n_samples = len(sources)
    if chunk_size is None:
        chunk_size = position_chunk_size(
            n_samples,
            config.DEFAULT_ITEMSIZE,
            config.TARGET_BYTES,
            config.MIN_POSITION_CHUNK_SIZE,
        )

    # Import data into the store
    logger.info(
        "Importing %i data sources with chunks of %i positions x %i samples (%.0f MB)",
        n_samples,
        chunk_size,
        n_samples,
        chunk_size * n_samples * config.DEFAULT_ITEMSIZE / 1e6,
    )
    try:
        pbzarr.import_d4(
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
        raise
