import re
from pathlib import Path

import pbzarr

from pbview.logging import cli_logger as logger


def ingest(
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
