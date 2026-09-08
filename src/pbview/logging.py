"""Custom logging setup with colorized output and filtering."""

import logging
import sys
from typing import List, Optional

import click


class DefaultFilter:
    def __init__(self, exclude: Optional[List[str]] = None):
        if exclude is None:
            exclude = []
        self.exclude = exclude

    def filter(self, record: logging.LogRecord) -> bool:
        _filter = True
        for ex in self.exclude:
            if record.name == "root":
                if record.getMessage().startswith("Dropping a patch"):
                    _filter = False
                    break
            elif record.name.startswith(ex):
                _filter = False
                break
        return _filter


class ColorizedTextHandler(logging.StreamHandler):
    """Custom colorized logging handler."""

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[%dm"
    BOLD_SEQ = "\033[1m"

    colors = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    def __init__(
        self,
        nocolor=False,
        stream=sys.stderr,
        formatter: Optional[logging.Formatter] = None,
        filt: Optional[logging.Filter] = None,
    ):
        super().__init__(stream=stream)
        if formatter:
            self.setFormatter(formatter)
        if filt:
            self.addFilter(filt)

    def emit(self, record):
        """Emit a message with custom formatting and color."""
        try:
            formatted_message = self.format(record)
            if formatted_message == "None" or formatted_message == "":
                return
            self.stream.write(self.decorate(record, formatted_message))
            self.stream.write(getattr(self, "terminator", "\n"))
            self.flush()
        except Exception:
            self.handleError(record)

    def decorate(self, record, message):
        """Add color to the log message"""
        message = [message]
        _, level = record.__dict__.get("event", None), record.levelname

        if level in self.colors:
            color = self.colors[level]
            message.insert(0, self.COLOR_SEQ % (30 + color))
            message.append(self.RESET_SEQ)

        return "".join(message)


formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
default_filter = DefaultFilter(
    exclude=[
        "bokeh",
        "bokeh.server",
        "bokeh.util.warnings",
        "panel",
        "requests",
        "tornado",
        "root.apply_json_patch",
        "root",
    ]
)
handler = ColorizedTextHandler(formatter=formatter, filt=default_filter)


def log_level(expose_value=False):
    """Setup logging level.

    Parameters:
        expose_value (bool): Whether to expose the value to the command function.
    """

    def callback(ctx, param, value):
        no_log_filter = ctx.params.get("no_log_filter", False)
        exclude = default_filter.exclude
        if no_log_filter:
            exclude = []
        loggers = ["pbview-app", "pbview-cli"]
        for logname in loggers:
            logger = logging.getLogger(logname)
            logger.setLevel(value)
            if len(logger.handlers) == 0:
                logger.addHandler(handler)
                logger.propagate = False
        for ex in exclude:
            logger = logging.getLogger(ex)
            logger.setLevel("CRITICAL")
            logger.addFilter(default_filter)
        return

    return click.option(
        "--log-level",
        default="INFO",
        help="Logging level",
        callback=callback,
        expose_value=expose_value,
        is_eager=False,
    )


app_logger = logging.getLogger("pbview-app")
cli_logger = logging.getLogger("pbview-cli")
