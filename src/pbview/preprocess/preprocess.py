from pathlib import Path

from dask.diagnostics import ProgressBar

from pbview.logging import app_logger as logger
from pbview.model.datastore import DataStore

from .track import compute_track_missingness, compute_track_sum


def preprocess(
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
