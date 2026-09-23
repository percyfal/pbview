from contextlib import nullcontext
from pathlib import Path

from dask.diagnostics import ProgressBar

from pbview.logging import cli_logger as logger
from pbview.model.datastore import DataStore


def generate_masks(
    path: Path | str,
    *,
    threshold: Path | str,
    output: Path | str | None = None,
    track: str = "depth",
    progress: bool = True,
    sampleinfo: Path | str | None = None,
) -> None:
    """Generate masks for the pbzarr store."""
    ds = DataStore(path=path, sampleinfo=sampleinfo)
    logger.info("Creating mask for track %s based on threshold %s", track, threshold)
    logger.info("Writing mask to pbzarr store at %s", output)
    print(ds)

    show_progress = ProgressBar() if progress else nullcontext()
    with show_progress:
        pass
    #     ds.create_mask(
    #         track=ds.tracks[track],
    #         threshold=threshold,
    #         workers=workers,
    #         progress=progress,
    #     )
