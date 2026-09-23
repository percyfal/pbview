from pathlib import Path

import click
from dask.diagnostics import ProgressBar

from pbview.logging import app_logger as logger
from pbview.logging import log_level
from pbview.model.datastore import DataStore
from pbview.preprocess.track import compute_track_missingness, compute_track_sum

from ._common import (
    chunk_size_option,
    path_argument,
    progress_option,
    sampleinfo_option,
    track_name_option,
    workers_option,
)


@click.command()
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
@sampleinfo_option()
def preprocess(
    path: Path | str,
    track_name: str,
    workers: int,
    progress: bool,
    chunk_size: int,
    sampleinfo: str | None = None,
) -> None:
    """Preprocess the pbzarr store for faster access."""
    run_preprocess(
        path,
        track=track_name,
        workers=workers,
        progress=progress,
        chunk_size=chunk_size,
        sampleinfo=sampleinfo,
    )


def run_preprocess(
    path: Path | str,
    track: str = "depth",
    workers: int = 1,
    progress: bool = True,
    chunk_size: int = 1_000_000,
    sampleinfo: Path | str | None = None,
    missing_cutoff: int = 3,
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
        track=ds.tracks[track], base_coord=ds.base_coord, missing_cutoff=missing_cutoff
    )
    logger.info("Writing track missingness to pbzarr store at %s", path)
    with ProgressBar():
        ds_miss.to_zarr(
            path,
            group=f"{track}_missing_cutoff={missing_cutoff}",
            mode="w",
        )
