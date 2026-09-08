from pathlib import Path

from pbview.datastore import DataStore
from pbview.logging import app_logger as logger


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
