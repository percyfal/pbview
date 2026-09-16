import re
from pathlib import Path

import pbzarr
from dask.diagnostics import ProgressBar

from pbview.datastore import DataStore
from pbview.logging import app_logger as logger
from pbview.preprocess.track import compute_track_missingness, compute_track_sum


def summarize(
    path: Path,
    *,
    sampleinfo: Path | str | None = None,
    annotation: Path | str | None = None,
    max_bins: int = 1_000,
    threads: int = 1,
    workers: int = 1,
    trackname: str = "depth",
):
    """Summarize a pbzarr store for visualization.

    Summarize data in pbzarr store for visualization, including
    binning and annotation.

    Args:
        path: The path to the pbzarr store.
        annotation: Path to an annotation file to include in the store.
        sampleinfo: Sampleinfo file mapping samples to populations
        max_bins: Maximum number of bins to use for summarization. Default is 1,000.
        threads: Number of threads to use for summarization. Default is 1.
        workers: Number of workers to use for importing data. Default is 1.
        trackname: Name of the track to summarize. Default is "depth".
    """
    pbzstore = DataStore(path)
    logger.info(pbzstore)
    # track = pbzstore.tracks[trackname]
    # tv = TrackView(track)


def import_d4(
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


def preprocess(
    path: Path | str,
    track: str = "depth",
    workers: int = 1,
    progress: bool = True,
    chunk_size: int = 1_000_000,
    sampleinfo: Path | str | None = None,
    missingness_threshold: int = 3,
) -> None:
    """Preprocess the pbzarr store for faster access.

    Add track_sum and track_count tracks for all samplesets.
    """
    ds = DataStore(path=path, sampleinfo=sampleinfo)
    ds_sum = compute_track_sum(track=ds.tracks[track], base_coord=ds.base_coord)
    logger.info("Writing track sum to pbzarr store at %s", path)
    with ProgressBar():
        ds_sum.to_zarr(path, group=f"{track}_sum", mode="w")

    ds_miss = compute_track_missingness(
        track=ds.tracks[track],
        base_coord=ds.base_coord,
        threshold=missingness_threshold,
    )
    logger.info("Writing track missingness to pbzarr store at %s", path)
    with ProgressBar():
        ds_miss.to_zarr(
            path,
            group=f"{track}_missingness_threshold={missingness_threshold}",
            mode="w",
        )
