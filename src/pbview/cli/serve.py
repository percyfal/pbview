import click

from pbview.logging import log_level

from ._common import (
    annotation_file_option,
    chunk_size_option,
    dashboard_option,
    dask_port_option,
    log_filter_option,
    max_positions_option,
    path_argument,
    port_option,
    setup_dask_scheduler,
    show_option,
    threads_option,
    use_dask_option,
    workers_option,
)


@click.command()
@path_argument(dir_okay=True, nargs=1)
@annotation_file_option()
@port_option()
@dask_port_option()
@show_option()
@threads_option()
@workers_option()
@dashboard_option()
@chunk_size_option()
@max_positions_option()
@use_dask_option()
@log_filter_option()
@log_level()
@click.option("--servable", is_flag=True, default=False, help="Make app servable")
@setup_dask_scheduler
def serve(
    path,
    port,
    annotation_file,
    show,
    servable,
    chunk_size,
    max_positions,
):
    """Serve the app.

    The default thresholds are estimated from at most --max-positions
    evenly spaced positions (0: all).
    """
    from ..app import serve as app_serve

    app_serve(
        path=path,
        port=port,
        show=show,
        servable=servable,
        verbose=False,
        chunk_size=chunk_size,
        max_positions=max_positions,
        annotation_file=annotation_file,
    )
