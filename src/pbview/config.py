"""Configuration settings.

This file stores configurations for the entire application such as figure
dimensions and color schemes.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-01"


RAW_CSS = """
        .sidenav#sidebar {
            background-color: WhiteSmoke;
        }
        .title {
            font-size: var(--type-ramp-plus-2-font-size);
        }
    """
DEFAULT_PARAMS = {
    "site": "pbview",
    "theme_toggle": False,
}
SIDEBAR_WIDTH = 300

# Global plot settings
PLOT_WIDTH = 1000
PLOT_HEIGHT = 600
THRESHOLD = 1000  # max number of points to overlay on a plot
PLOT_COLOURS = ["#15E3AC", "#0FA57E", "#0D5160"]

# VCard settings
SIDEBAR_BACKGROUND = "#5CB85D"
VCARD_STYLE = {
    "background": "WhiteSmoke",
}

# Sample set settings
DEFAULT_SAMPLE_SET = "ALL"

# Chunking
MIN_POSITION_CHUNK_SIZE = 1_000_000
TARGET_BYTES = 100_000_000

# Schema
SCHEMA_VERSION = 1
