import atexit
import contextlib
import functools
import multiprocessing  # noqa
import sys
from typing import Callable, Mapping  # noqa

import click
import dask
from click.decorators import FC
from dask.distributed import Client, LocalCluster

from pbview import config
from pbview.logging import app_logger as logger  # noqa
from pbview.model.track import MissingSummaryError


def log_filter_option(expose_value: bool = False) -> Callable[[FC], FC]:
    """Disable logging filters"""
    return click.option(
        "--no-log-filter",
        default=False,
        is_flag=True,
        expose_value=expose_value,
        help="Do not filter the output log (advanced debugging only)",
    )


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
    return click.argument("d4files", type=click.Path(exists=True), nargs=nargs)


def sampleset_info_option() -> Callable[[FC], FC]:
    return click.option(
        "--sampleset-info",
        "--sampleset_info",  # deprecated spelling, kept for existing scripts
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
        help="Number of threads / threads per worker",
        type=click.IntRange(1, multiprocessing.cpu_count()),
    )


def workers_option(default: int = 1) -> Callable[[FC], FC]:
    return click.option(
        "--workers",
        default=default,
        help="Number of workers to use for pre-processing",
        type=click.IntRange(1, multiprocessing.cpu_count()),
    )


def memory_limit_option(default: str = "4GB") -> Callable[[FC], FC]:
    return click.option(
        "--memory-limit", default=default, help="Memory limit per worker", type=str
    )


def dashboard_option(default: int = 44446) -> Callable[[FC], FC]:
    return click.option(
        "--dashboard", default=default, help="Port to serve dashboard on", type=int
    )


def use_dask_option(default: bool = False) -> Callable[[FC], FC]:
    return click.option(
        "--use-dask/--no-use-dask",
        default=default,
        help="Use dask for multiprocessing. Provides dashboard",
        type=bool,
    )


def missing_data_threshold_option(
    multiple: bool = False,
    default: int | tuple[int, ...] | None = 3,
    deprecated_alias: bool = False,
) -> Callable[[FC], FC]:
    """Coverage at or below the threshold calls a base missing."""
    names = ["--missing-data-threshold"]
    if deprecated_alias:
        names.append("--threshold")  # old spelling
    return click.option(
        *names,
        "missing_data_threshold",
        type=click.IntRange(min=0),
        multiple=multiple,
        default=default,
        show_default=default not in (None, ()),
        help="Coverage at or below this value calls a base missing"
        + (" (repeatable)" if multiple else ""),
    )


@contextlib.contextmanager
def missing_summary_errors():
    """Report a missing summary track as a clean command line error."""
    try:
        yield
    except MissingSummaryError as e:
        raise click.ClickException(str(e)) from e


def max_positions_option(default: int | None = None) -> Callable[[FC], FC]:
    """Cap on the positions used to estimate the threshold defaults."""
    return click.option(
        "--max-positions",
        type=click.IntRange(min=0),
        default=default,
        show_default=f"{config.MAX_POSITIONS:,}" if default is None else True,
        help=(
            "Estimate the default thresholds from at most this many evenly "
            "spaced positions; 0 uses all positions. Env: PBVIEW_MAX_POSITIONS"
        ),
    )


def dask_port_option(default: int = 18786) -> Callable[[FC], FC]:
    return click.option("--dask-port", default=default, help="Port to serve on")


def max_bins_option(default: int = 1000) -> Callable[[FC], FC]:
    return click.option(
        "--max-bins", default=default, help="Maximum number of bins to display"
    )


def chunk_size_option(default: int = 1000000) -> Callable[[FC], FC]:
    return click.option(
        "--chunk-size",
        default=default,
        help=(
            "Chunk size for processing large files. Consider "
            "reducing if number of samples is large (n>500)."
        ),
    )


def port_option(default: int = 5507) -> Callable[[FC], FC]:
    return click.option("port", "--port", default=default, help="Port to serve on")


def show_option(default: bool = False) -> Callable[[FC], FC]:
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


def _init_dask(
    port=18786,
    dashboard=44446,
    threads=1,
    workers=8,
    memory_limit="4GB",
) -> Client:
    """Initialize dask scheduler"""
    try:
        client = Client(f"tcp://localhost:{port}", timeout="2s")
        logger.info("Connected to existing dask cluster at :%d", port)
    except OSError:
        cluster = LocalCluster(
            processes=False,
            scheduler_port=port,
            dashboard_address=f":{dashboard}",
            threads_per_worker=threads,
            n_workers=workers,
            memory_limit=memory_limit,
        )
        client = Client(cluster)
        logger.info("Started new dask cluster; dashboard at :%d", dashboard)
        atexit.register(cluster.close)
    atexit.register(client.close)
    return client


def _init_threads(workers=2, threads=2, memory_limit="4GB", **_):
    logger.info(
        "Setting threads to %d, workers to %d, memory limit to %s",
        threads,
        workers,
        memory_limit,
    )
    dask.config.set(
        scheduler="threads",
        num_workers=workers,
        threads_per_worker=threads,
        memory_limit=memory_limit,
    )


def setup_dask_scheduler(func):
    @functools.wraps(func)
    def wrapper(**kwargs):
        kw = {}
        kw["threads"] = kwargs.pop("threads", 2)
        kw["workers"] = kwargs.pop("workers", 2)
        kw["memory_limit"] = kwargs.pop("memory_limit", "4GB")
        kw["dask_port"] = kwargs.pop("dask_port", 18786)
        kw["dashboard"] = kwargs.pop("dashboard", 44446)
        use_dask = kwargs.pop("use_dask", False)
        if use_dask:
            _init_dask(**kw)
        else:
            _init_threads(**kw)
        return func(**kwargs)

    return wrapper


def set_threads(func):
    @functools.wraps(func)
    def wrapper(**kwargs):
        kw = {}
        kw["threads"] = kwargs.pop("threads", 2)
        kw["workers"] = kwargs.pop("workers", 1)
        kw["memory_limit"] = kwargs.pop("memory_limit", "4GB")
        use_dask = kwargs.pop("use_dask", False)
        if use_dask:
            logger.error(
                "Dask is not supported for console. "
                "Please use the --no-use-dask option."
            )
            sys.exit(1)
        else:
            _init_threads(**kw)
        return func(**kwargs)

    return wrapper


def sampleset_name_option(default: str = "ALL") -> Callable[[FC], FC]:
    return click.option(
        "--sampleset-name",
        type=str,
        default=default,
        help="Name of the sample set to use for analysis",
    )
