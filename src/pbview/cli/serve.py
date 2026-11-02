import click

from pbview.logging import log_level

from ._common import (
    address_option,
    annotation_file_option,
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
    websocket_origin_option,
    workers_option,
)


@click.command()
@path_argument(dir_okay=True, nargs=1)
@annotation_file_option()
@port_option()
@address_option()
@websocket_origin_option()
@dask_port_option()
@show_option()
@threads_option()
@workers_option()
@dashboard_option()
@max_positions_option()
@use_dask_option()
@log_filter_option()
@log_level()
@click.option("--servable", is_flag=True, default=False, help="Make app servable")
@setup_dask_scheduler
def serve(
    path,
    port,
    address,
    allow_websocket_origin,
    annotation_file,
    show,
    servable,
    max_positions,
):
    """Serve the app.

    The default thresholds are estimated from at most --max-positions
    evenly spaced positions (0: all).

    To view the app from another machine, for example when it runs on a
    compute node, listen on an address the machine can be reached on
    (--address 0.0.0.0) and forward the port with ssh, e.g.
    `ssh -L 8080:NODE:8080 user@login-node`, then open http://localhost:8080.
    """
    from ..app import serve as app_serve

    app_serve(
        path=path,
        port=port,
        address=address,
        websocket_origin=list(allow_websocket_origin) or None,
        show=show,
        servable=servable,
        verbose=False,
        max_positions=max_positions,
        annotation_file=annotation_file,
    )
