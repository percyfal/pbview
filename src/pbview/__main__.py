import click
import pandas as pd  # noqa
import panel as pn  # noqa

from pbview.logging import app_logger as logger  # noqa
from pbview.logging import log_level  # noqa
from pbview.model import datastore  # noqa

from . import __version__
from .cli import import_d4, mask, preprocess, summarize


@click.group()
@click.version_option(version=__version__)
def cli():
    """Command line interface for pbview."""


cli.add_command(import_d4.import_d4)
cli.add_command(preprocess.preprocess)
cli.add_command(summarize.summarize)
cli.add_command(mask.mask)


if __name__ == "__main__":
    cli()
