"""Configuration settings.

This file stores configurations for the entire application such as figure
dimensions and color schemes.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-01"

import os

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

# Grouping set settings
DEFAULT_GROUPING_SET = "ALL"

# Summary track provenance, written by preprocess and used to find summaries
SUMMARY_SUM = "preprocess_sum"
SUMMARY_MISSINGNESS = "preprocess_missingness"

# Chunking

MIN_POSITION_CHUNK_SIZE = int(os.environ.get("PBVIEW_MIN_POSITION_CHUNK_SIZE", 100_000))
TARGET_BYTES: int = int(os.environ.get("PBVIEW_TARGET_BYTES", 100_000_000))
# Bytes per value of an imported track (pbzarr stores d4 coverage as int32)
DEFAULT_ITEMSIZE: int = 4
MAX_COVERAGE_BIN: int = int(os.environ.get("PBVIEW_MAX_COVERAGE_BIN", 100_000))
MAX_COVERAGE_PER_SAMPLE: int = int(
    os.environ.get("PBVIEW_MAX_COVERAGE_PER_SAMPLE", 100)
)
# Threshold defaults are estimated from at most this many evenly spaced
# positions (0: use all positions)
MAX_POSITIONS: int = int(os.environ.get("PBVIEW_MAX_POSITIONS", 50_000_000))
# PCA of coverage uses the mean coverage of windows of this many positions
PCA_WINDOW_SIZE: int = int(os.environ.get("PBVIEW_PCA_WINDOW_SIZE", 10_000))


# FIXME (pbzarr>0.6): the spatial dimensions will be accessible from
# the root node
# Spatial dimensions
SPATIAL_DIMS = ["position", "contig", "contig_boundary"]

# Schema
SCHEMA_VERSION = 1
