import multiprocessing  # noqa
from typing import Callable, Mapping  # noqa

import atexit

import sys
import click
import dask
from dask.distributed import Client, LocalCluster
import functools
from click.decorators import FC

from pbview.logging import app_logger as logger  # noqa


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
    return click.argument("d4", type=click.Path(exists=True), nargs=nargs)


# FIXME: rename throughout to sampleset_info: this is about grouping
# samples into sample sets
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


def threshold_option(default: int = 3) -> Callable[[FC], FC]:
    return click.option(
        "--threshold",
        default=default,
        help="Coverage threshold for calling a base as present",
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
        kw["port"] = kwargs.pop("port", 18786)
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
