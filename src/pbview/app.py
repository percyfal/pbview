"""Main application for pbview.

Provides the App class that is the main application for pbview.
The App subclasses the Viewer class from panel and renders a
panel.FastListTemplate.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"

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


def serve(servable, **kw):
    """Serve the app"""
    logger.info("Serving main app")

    kwargs = {"path": kw.pop("path", None), "sampleinfo": kw.pop("sampleinfo", None)}
    ds = DataStore(**kwargs)
    dsview = DataStoreView(datastore=ds)
    app_ = App(datastoreview=dsview)

    if servable:
        return app_.view().servable()
    return pn.serve(app_.view(), **kw)
