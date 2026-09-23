import click

from pbview.logging import log_level

from ._common import (
    annotation_file_option,
    chunk_size_option,
    dashboard_option,
    dask_port_option,
    log_filter_option,
    path_argument,
    port_option,
    sampleinfo_option,
    show_option,
    threads_option,
    use_dask_option,
    workers_option,
)


@click.command()
@path_argument(dir_okay=True, nargs=1)
@annotation_file_option()
@sampleinfo_option()
@port_option()
@dask_port_option()
@show_option()
@threads_option()
@workers_option()
@dashboard_option()
@chunk_size_option()
@use_dask_option()
@log_filter_option()
@log_level()
@click.option("--servable", is_flag=True, default=False, help="Make app servable")
def serve(
    path,
    annotation_file,
    sampleinfo,
    port,
    dask_port,
    show,
    threads,
    servable,
    workers,
    dashboard,
    chunk_size,
    use_dask,
):
    """Serve the app."""
    from ..app import serve as app_serve

    app_serve(
        path=path,
        port=port,
        dask_port=dask_port,
        show=show,
        threads_per_worker=threads,
        n_workers=workers,
        servable=servable,
        sampleinfo=sampleinfo,
        dashboard=dashboard,
        verbose=False,
        chunk_size=chunk_size,
        use_dask=use_dask,
    )
