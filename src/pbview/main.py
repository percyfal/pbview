"""Helper module for serving the pbview app from the command line
using panel serve.

This module is used to serve the pbview app from the command line
using panel serve. One use case is for development purposes where the
--dev argument enables automated reloading of the app when the source
code changes. To launch the app from the command line run:

$ panel serve --dev --admin --show --args path/to/sum.d4
  --annotation-file path/to/annotation.gff3

See https://panel.holoviz.org/how_to/server/commandline.html for more
information.
"""

from pbview import app  # noqa
from pbview import datastore  # noqa

import sys
from collections import deque
from pbview.__main__ import serve, preprocess
from pbview.logging import app_logger as logger

arglist = deque(sys.argv)
arglist.popleft()

try:
    argfun = arglist.popleft()
except IndexError:
    argfun = "serve"

if argfun == "serve":
    arglist.append("--servable")
    fun = serve
elif argfun == "preprocess":
    fun = preprocess
else:
    fun = serve
    arglist.append("--help")

try:
    fun(arglist, standalone_mode=False)
except Exception as e:
    logger.error(e)
    sys.exit(1)
