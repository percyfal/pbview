from pathlib import Path

import click

from pbview.logging import app_logger as logger
from pbview.logging import log_level
from pbview.model.datastore import DataStore

from ._common import (
    annotation_file_option,
    log_filter_option,
    max_bins_option,
    path_argument,
    sampleinfo_option,
    threads_option,
    workers_option,
)


@click.command()
@path_argument(dir_okay=True, nargs=1)
@annotation_file_option()
@sampleinfo_option()
@threads_option()
@workers_option()
@max_bins_option()
@log_filter_option()
@log_level()
def summarize(
    path: Path,
    sampleinfo: Path | str | None,
    annotation_file: Path | str | None,
    threads: int,
    workers: int,
    max_bins: int,
):
    """Run summary analysis on a pbzarr store."""
    if annotation_file is not None:
        annotation_file = Path(annotation_file)

    logger.info("Running summary analysis on %s", path)

    data = run_summarize(
        path,
        annotation=annotation_file,
        sampleinfo=sampleinfo,
        max_bins=max_bins,
        threads=threads,
        workers=workers,
    )
    logger.info(data)


def run_summarize(
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
