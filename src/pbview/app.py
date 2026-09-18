"""Main application for pbview.

Provides the App class that is the main application for pbview.
The App subclasses the Viewer class from panel and renders a
panel.FastListTemplate.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-01"

import atexit

import dask
import panel as pn
import param
from dask.distributed import Client, LocalCluster
from panel.viewable import Viewer

from pbview import config
from pbview.logging import app_logger as logger
from pbview.model.datastore import DataStore
from pbview.view.datastore import DataStoreView

pn.extension("vega", throttled=True)
pn.extension(sizing_mode="stretch_width")
pn.extension("tabulator")


def _init_dask(
    port=18786, dashboard=44446, threads_per_worker=1, n_workers=8
) -> Client:
    try:
        client = Client(f"tcp://localhost:{port}", timeout="2s")
        logger.info("Connected to existing dask cluster at :%d", port)
    except OSError:
        cluster = LocalCluster(
            processes=False,
            scheduler_port=port,
            dashboard_address=f":{dashboard}",
            threads_per_worker=threads_per_worker,
            n_workers=n_workers,
        )
        client = Client(cluster)
        logger.info("Started new dask cluster; dashboard at :%d", dashboard)
        atexit.register(cluster.close)
    atexit.register(client.close)
    return client


class App(Viewer):
    datastoreview = param.ClassSelector(class_=DataStoreView)

    def __init__(self, **params):
        super().__init__(**params)
        self.title = str(self.datastoreview.title)

    def view(self):
        dsv = self.datastoreview
        self._template = pn.template.FastListTemplate(
            title=dsv.title,
            header=[dsv.chooser()],
            sidebar=[dsv.sidebar()],
            main=[dsv.main],
            sidebar_width=config.SIDEBAR_WIDTH,
        )
        return self._template


def serve(servable, **kw):
    """Serve the app"""
    kwargs = {
        "port": kw.pop("dask_port", 18786),
        "dashboard": kw.pop("dashboard", 44446),
        "threads_per_worker": kw.pop("threads_per_worker", 2),
        "n_workers": kw.pop("n_workers", 2),
    }
    if kw.pop("use_dask", False):
        _ = _init_dask(**kwargs)
    else:
        dask.config.set(scheduler="threads", num_workers=kwargs["n_workers"])
    logger.info("Serving main app")

    kwargs = {"path": kw.pop("path", None), "sampleinfo": kw.pop("sampleinfo", None)}
    ds = DataStore(**kwargs)
    dsview = DataStoreView(datastore=ds)
    app_ = App(datastoreview=dsview)

    if servable:
        return app_.view().servable()
    return pn.serve(app_.view(), **kw)
