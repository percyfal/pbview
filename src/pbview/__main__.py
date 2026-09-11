import multiprocessing  # noqa
from pathlib import Path  # noqa
from typing import Callable, Mapping  # noqa

import click
import pandas as pd  # noqa
import panel as pn  # noqa
from click.decorators import FC

from pbview import (
    # cache,  # noqa
    cli as pbview_cli,  # noqa
    datastore,  # noqa
)

# from pbview.d4utils import commands as d4utils_cmd  # noqa
# from pbview.model import d4  # noqa
from pbview.logging import log_level  # noqa
from pbview.logging import app_logger as logger  # noqa

from . import (
    __version__,  # noqa
    app,  # noqa
)


def log_filter_option(expose_value: bool = False) -> Callable[[FC], FC]:
    """Disable logging filters"""
    return click.option(
        "--no-log-filter",
        default=False,
        is_flag=True,
        expose_value=expose_value,
        help="Do not filter the output log (advanced debugging only)",
    )


# def cachedir_option() -> Callable[[FC], FC]:
#     return click.option(
#         "--cachedir",
#         default=cache.CACHEDIR,
#         expose_value=True,
#         help="Set the cache dir",
#     )


def path_argument(
    exists: bool = True, dir_okay: bool = False, nargs: int = 1, required: bool = True
) -> Callable[[FC], FC]:
    return click.argument(
        "path",
        type=click.Path(exists=exists, dir_okay=dir_okay),
        nargs=nargs,
        required=required,
    )


def d4_argument(nargs: int = 1) -> Callable[[FC], FC]:
    return click.argument("d4", type=click.Path(exists=True), nargs=nargs)


def sampleinfo_option() -> Callable[[FC], FC]:
    return click.option(
        "--sampleinfo",
        type=click.Path(exists=True, dir_okay=False),
        help=(
            "2-column tab-separated sample info file with sample to population mapping"
        ),
    )


def region_argument(
    exists: bool = True, dir_okay: bool = False, nargs: int = 1
) -> Callable[[FC], FC]:
    return click.argument(
        "region",
        type=click.Path(exists=exists, dir_okay=dir_okay),
        nargs=nargs,
    )


def annotation_file_option(default: str = None) -> Callable[[FC], FC]:
    return click.option(
        "--annotation-file",
        default=default,
        type=click.Path(exists=True),
        help="Annotation file in gff format",
    )


def threads_option(default: int = 1) -> Callable[[FC], FC]:
    return click.option(
        "--threads",
        default=default,
        help="Number of threads per worker to use for pre-processing",
        type=click.IntRange(1, multiprocessing.cpu_count()),
    )


def workers_option(default: int = 1) -> Callable[[FC], FC]:
    return click.option(
        "--workers",
        default=default,
        help="Number of workers to use for pre-processing",
        type=click.IntRange(1, multiprocessing.cpu_count()),
    )


def threshold_option(default: int = 3) -> Callable[[FC], FC]:
    return click.option(
        "--threshold",
        default=default,
        help="Coverage threshold for calling a base as present",
    )


def max_bins_option(default: int = 1000) -> Callable[[FC], FC]:
    return click.option(
        "--max-bins", default=default, help="Maximum number of bins to display"
    )


def port_option(default: int = 8080) -> Callable[[FC], FC]:
    return click.option("--port", default=default, help="Port to serve on")


def show_option(default: bool = True) -> Callable[[FC], FC]:
    return click.option(
        "--show/--no-show",
        default=default,
        help="Launch a web-browser showing the app",
    )


def progress_option(default: bool = True) -> Callable[[FC], FC]:
    return click.option(
        "-p",
        "--progress/--no-progress",
        default=default,
        help="Show progress bar for long-running tasks",
    )


def dryrun_option(default: bool = False) -> Callable[[FC], FC]:
    return click.option(
        "-n",
        "--dry-run/--no-dry-run",
        default=default,
        help=(
            "Dry run: do not actually perform the action, just print what would be done"
        ),
    )


def track_name_option(default: str = "depth") -> Callable[[FC], FC]:
    return click.option(
        "-t", "--track-name", default=default, help="Set track name for import"
    )


@click.group()
@click.version_option(version=__version__)
def cli():
    """Command line interface for pbview."""


@cli.command("import")
@path_argument(exists=False, dir_okay=True, nargs=1)
@d4_argument(nargs=-1)
@workers_option(default=1)
@track_name_option()
@log_level()
@progress_option()
def import_d4(
    path: Path | str,
    d4: list[Path] | list[str],
    track_name: str,
    workers: int,
    progress: bool,
):
    """Import d4 files into a pbzarr store."""
    if len(d4) == 0:
        logger.error("Provide at least one d4 source file to import")
        return
    datastore.import_d4(
        path,
        d4=d4,
        track=track_name,
        workers=workers,
        progress=progress,
    )


@cli.command()
@path_argument(dir_okay=True, nargs=1)
@annotation_file_option()
@sampleinfo_option()
@port_option()
@show_option()
@threads_option()
@log_filter_option()
@log_level()
# @cachedir_option()
@click.option("--summarize", is_flag=True, default=False, help="Run summarize analysis")
@click.option("--servable", is_flag=True, default=False, help="Make app servable")
def serve(path, annotation_file, sampleinfo, port, show, threads, servable, summarize):
    """Serve the app."""
    app.serve(
        path=path,
        port=port,
        show=show,
        threads=threads,
        servable=servable,
        sampleinfo=sampleinfo,
        # cachedir=cachedir,
        verbose=False,
        summarize=summarize,
    )


@cli.command()
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

    data = pbview_cli.summarize(
        path,
        annotation=annotation_file,
        sampleinfo=sampleinfo,
        max_bins=max_bins,
        threads=threads,
        workers=workers,
    )
    logger.info(data)


# FIXME: This is a placeholder for the preprocess command. It should
# be implemented to handle preprocessing of expensive data before
# serving. The user should be able to provide
#   - a grid of maxbin values for the histograms per sample-set;
#     defaults to mean + 3-5stddev rounded to next 100? or base on
#     magnitude of the mean?
#   - a grid of threshold values for the histograms per sample-set; default 0-5?
#   - sampleinfo such that histograms are pre-computed for all sampleset groupings
#   - contig length filters to start with sensible contigs (exclude low-complexity)
@cli.command()
def preprocess():
    pass


if __name__ == "__main__":
    cli()
