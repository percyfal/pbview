"""Main application for pbview.

Provides the App class that is the main application for pbview.
The App subclasses the Viewer class from panel and renders a
panel.FastListTemplate.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"

import ipaddress

import panel as pn
import param
from panel.viewable import Viewer

from pbview import config
from pbview.logging import app_logger as logger
from pbview.model.datastore import DataStore
from pbview.view.datastore import DataStoreView

pn.extension("vega", throttled=True)
pn.extension(sizing_mode="stretch_width")
pn.extension("tabulator")


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


def is_loopback(address: str | None) -> bool:
    """Whether `address` can only be reached from the machine itself."""
    if address is None or address == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:  # a host name
        return False


def serve(servable, **kw):
    """Serve the app

    The data store is opened once and shared by all sessions (it is not
    modified, and its caches make later sessions start quickly). Every
    session gets its own views and selection state: a Bokeh model can
    only belong to one document, so views built once and served to every
    session break when a session ends (the contig plot came up empty).
    """
    logger.info("Serving main app")

    ds = DataStore(path=kw.pop("path", None))
    max_positions = kw.pop("max_positions", None)

    def create():
        dsview = DataStoreView(datastore=ds, max_positions=max_positions)
        return App(datastoreview=dsview).view()

    if servable:
        return create().servable()
    # Panel listens on all interfaces without an address; the app has no login
    address = kw["address"] = kw.get("address") or "localhost"
    port = kw.get("port")
    origins = kw.get("websocket_origin")
    if origins and f"localhost:{port}" not in origins:
        kw["websocket_origin"] = [*origins, f"localhost:{port}"]
    logger.info("Serving at http://%s:%s", address, port)
    if not is_loopback(address):
        logger.warning(
            "Listening on %s: the app has no login, so everyone who can reach "
            "this address and port can use it and read the data",
            address,
        )
    return pn.serve(create, **kw)
