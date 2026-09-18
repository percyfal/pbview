"""
Summary page view
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-18"


import hvplot.pandas  # noqa
import panel as pn


pn.extension("tabulator")


class SummaryPage(pn.viewable.Viewer):
    def __init__(self, coverage_page, missingness_page, **params):
        super().__init__(**params)
        self._cov_table = coverage_page.table
        self._mis_table = missingness_page.table

    def __panel__(self):
        return pn.Column(
            "# Summary",
            "## Coverage",
            self._cov_table,
            "## Missingness",
            self._mis_table,
        )
