"""
Summary page view
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-18"

import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool, CDSView, GroupFilter
from bokeh.plotting import figure

from pbview.model.selection import SelectionStateBase
from pbview.model.track import Track

pn.extension("tabulator")


class SummaryPage(pn.viewable.Viewer):
    state = param.ClassSelector(class_=SelectionStateBase, is_instance=True)
    track = param.ClassSelector(class_=Track, is_instance=True)

    def __init__(self, coverage_page, missingness_page, **params):
        super().__init__(**params)
        self._suspend = True
        self._color_map = self.state.datastore.sample_set_colors
        self._source = ColumnDataSource(
            data=dict(length=[], mean=[], contig=[], color=[])
        )
        self._source.selected.on_change("indices", self._on_select)

        self._cov_table = coverage_page.table
        self._mis_table = missingness_page.table

        self.state.param.watch(lambda *_: self._refresh(), "coord")
        self.state.param.watch(lambda *_: self._refresh(), "contigs")
        self._refresh()
        self._figure = self._make_figure()
        self._pane = pn.pane.Bokeh(self._figure)
        self._export_btn = pn.widgets.FileDownload(
            label="Export thresholds",
            filename="thresholds.yml",
            callback=self._export_callback,
            button_type="primary",
        )

    def _export_callback(self):
        from io import StringIO

        from ..model.thresholds import profile_from_state

        profile = profile_from_state(
            self.state,
            datastore_id=self.state.datastore.id,
        )
        return StringIO(profile.to_yaml())

    def _make_figure(self):
        hover = HoverTool(
            tooltips=[
                ("Contig", "@contig"),
                ("Length", "@length"),
                ("Mean coverage", "@mean"),
                ("Total mean coverage", "@total_mean"),
                ("Sample set", "@sample_set"),
            ]
        )
        p = figure(
            x_axis_type="log",
            sizing_mode="stretch_width",
            height=400,
            tools=["box_select", "tap", "pan", "wheel_zoom", "reset", "save", hover],
            active_drag="box_select",
            title="Contig length vs mean coverage",
        )
        renderers = {}
        for s, color in self._color_map.items():
            view = CDSView(filter=GroupFilter(column_name="sample_set", group=s))
            r = p.scatter(
                "length",
                "mean",
                source=self._source,
                view=view,
                fill_color=color,
                line_color=color,
                size=12,
                legend_label=s,
            )
            renderers[s] = r
        return p

    def _refresh(self):
        self._suspend = True
        try:
            df = self._contig_scatter_df()  # includes color, alpha columns
            self._source.data = {c: df[c].to_numpy() for c in df.columns}
        finally:
            self._suspend = False

    def _on_select(self, attr, old, new):
        try:
            if self._suspend:
                return
            if not new:
                all_contigs = list(self.state.coord.contigs_all)
                if set(all_contigs) == set(self.state.contigs):
                    return
                self.state.contigs = all_contigs
                return
            contigs = sorted(set(self._source.data["contig"][i] for i in new))
            if set(contigs) == set(self.state.coord.contigs):
                return
            self.state.contigs = contigs

        except Exception:
            import traceback

            traceback.print_exc()

    def _contig_scatter_df(self):
        coord = self.state.coord
        names = coord.contigs_all
        lengths = coord.contig_len_all
        selected = np.asarray(~coord.contig_mask, dtype=np.int8) / 2 + 0.3
        dflist = []
        for s in coord.sample_set_names:
            means = self.track._pre.contig_mean_coverage(s)
            dflist.append(
                pd.DataFrame(
                    {
                        "sample_set": s,
                        "contig": names,
                        "length": lengths,
                        "mean": means / coord.with_sample_sets([s]).n_samples,
                        "selected": selected,
                        "total_mean": means,
                    }
                )
            )
        df = pd.concat(dflist)
        return df

    def __panel__(self):
        return pn.Column(
            "# Summary",
            self._export_btn,
            "## Contig summary",
            self._pane,
            "## Coverage",
            self._cov_table,
            "## Missingness",
            self._mis_table,
        )
