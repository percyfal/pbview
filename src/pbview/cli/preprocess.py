from pathlib import Path

import click

from pbview.logging import log_level
from pbview.preprocess import preprocess as preprocess_mod

from ._common import (
    chunk_size_option,
    memory_limit_option,
    missing_data_threshold_option,
    missing_summary_errors,
    path_argument,
    progress_option,
    sampleset_info_option,
    set_threads,
    threads_option,
    track_name_option,
    workers_option,
)


@click.group()
def preprocess() -> None:
    """Preprocess the Zarr store for faster access"""
    pass


@click.command("all")
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@threads_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
@sampleset_info_option()
@missing_data_threshold_option(multiple=True, default=(3,))
@set_threads
def preprocess_all(
    path: Path | str,
    track_name: str,
    progress: bool,
    chunk_size: int,
    missing_data_threshold: tuple[int, ...],
    sampleset_info: str | None = None,
) -> None:
    """Aggregate sum and missingness data."""
    preprocess_mod.preprocess(
        path,
        track=track_name,
        progress=progress,
        chunk_size=chunk_size,
        sampleset_info=sampleset_info,
        missing_cutoffs=missing_data_threshold,
    )


@click.command("sum")
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@threads_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
@sampleset_info_option()
@set_threads
def preprocess_sum(
    path: Path | str,
    track_name: str,
    progress: bool,
    chunk_size: int,
    sampleset_info: str | None = None,
) -> None:
    """Aggregate coverage sum data."""
    preprocess_mod.preprocess(
        path,
        track=track_name,
        progress=progress,
        chunk_size=chunk_size,
        sampleset_info=sampleset_info,
        preprocess_missingness=False,
    )


@click.command("missingness")
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@threads_option(default=1)
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
@sampleset_info_option()
@missing_data_threshold_option(multiple=True, default=(3,))
@set_threads
def preprocess_missingness(
    path: Path | str,
    track_name: str,
    progress: bool,
    chunk_size: int,
    missing_data_threshold: tuple[int, ...],
    sampleset_info: str | None = None,
) -> None:
    """Aggregate individual missingness data.

    Give --missing-data-threshold several times to summarize several
    thresholds in one run.
    """
    preprocess_mod.preprocess(
        path,
        track=track_name,
        progress=progress,
        chunk_size=chunk_size,
        sampleset_info=sampleset_info,
        missing_cutoffs=missing_data_threshold,
        preprocess_sum=False,
    )


@click.command("thresholds")
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@threads_option(default=1)
@memory_limit_option()
@track_name_option()
@log_level()
@progress_option()
@chunk_size_option()
@missing_data_threshold_option(default=3, deprecated_alias=True)
@set_threads
def preprocess_thresholds(
    path: Path | str,
    track_name: str,
    progress: bool,
    chunk_size: int,
    missing_data_threshold: int = 3,
) -> None:
    """Compute thresholds for missingness and sum data.

    The missingness summary for the chosen threshold must have been
    created with `pbview preprocess missingness`.
    """
    with missing_summary_errors():
        preprocess_mod.compute_thresholds(
            path,
            track_name=track_name,
            progress=progress,
            chunk_size=chunk_size,
            missing_cutoff=missing_data_threshold,
        )


preprocess.add_command(preprocess_all)
preprocess.add_command(preprocess_sum)
preprocess.add_command(preprocess_missingness)
preprocess.add_command(preprocess_thresholds)
