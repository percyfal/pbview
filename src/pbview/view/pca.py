"""
PCA page: the samples in the space of the first principal components
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-11-01"

from io import StringIO

import holoviews as hv
import pandas as pd
import panel as pn
import param
from panel.viewable import Viewer

from pbview import config
from pbview.model.pca import PcaResult
from pbview.model.selection import SelectionStateBase
from pbview.model.track import Track

pn.extension("tabulator")

MAX_COMPONENTS_TO_CHOOSE = 10


class PcaPage(Viewer):
    """Principal components of the coverage, with the samples as features.

    The PCA reads the whole selection once, which takes a while for a large
    genome, so it runs when the button is pressed, not at every change of the
    selection. The page says when the selection or the window size has
    changed since the last run.
    """

    state = param.ClassSelector(class_=SelectionStateBase, is_instance=True)
    track = param.ClassSelector(class_=Track, is_instance=True)
    window_size = param.Integer(
        default=config.PCA_WINDOW_SIZE,
        bounds=(1, None),
        doc="Positions per window: the mean coverage of each sample in "
        "windows of this size is what the PCA is made of",
    )
    x_component = param.Selector(default=1, objects=[1, 2], doc="Component on x")
    y_component = param.Selector(default=2, objects=[1, 2], doc="Component on y")
    show_labels = param.Boolean(default=False, doc="Write the sample names")
    result = param.ClassSelector(class_=PcaResult, default=None, allow_None=True)
    ran_for = param.Parameter(default=None, doc="Selection and window of `result`")
    error = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self._colors = self.state.datastore.sample_set_colors
        self._sets = self.track.group_membership()
        self._button = pn.widgets.Button(name="Run PCA", button_type="primary")
        self._button.on_click(self._on_click)
        self._download = pn.widgets.FileDownload(
            label="Download scores",
            filename="pca_scores.tsv",
            callback=self._scores_tsv,
            button_type="default",
        )

    # -- computing

    def _key(self):
        """The selection and window size that a run would use."""
        resolved = self.state.coord.resolve_for(self.track, self.state.datastore.genome)
        return resolved, self.window_size

    def run(self) -> None:
        """Run the PCA for the current selection and window size."""
        key = self._key()
        try:
            result = self.track.pca(*key)
        except ValueError as e:
            self.error = str(e)
            self.result = None
            self.ran_for = key
            return
        self.error = ""
        k = min(result.n_components, MAX_COMPONENTS_TO_CHOOSE)
        for selector in (self.param.x_component, self.param.y_component):
            selector.objects = list(range(1, k + 1))
        self.x_component, self.y_component = 1, min(2, k)
        self.result = result
        self.ran_for = key

    def _on_click(self, event) -> None:
        self._button.loading = True
        try:
            self.run()
        finally:
            self._button.loading = False

    # -- what is shown

    def _status(self, result, ran_for, coord, window_size):
        if ran_for is None:
            text = "Press **Run PCA** to compute the PCA of the selected samples."
        elif self.error:
            text = f"Not computed: {self.error}"
        elif self._key() != ran_for:
            text = (
                "The selection or the window size has changed since the last "
                "run: press **Run PCA** to update."
            )
        else:
            text = (
                f"PCA of {len(result.samples)} samples in {result.n_windows} "
                f"windows of {result.window_size:,} bp, each sample scaled by its "
                "mean coverage."
            )
        return pn.pane.Markdown(text)

    def _frame(self, result: PcaResult) -> pd.DataFrame:
        df = result.scores(sample_sets=self._sets)
        df["set"] = [self._sets[s][0] for s in df["sample"]]
        return df

    def _scatter(self, result, x, y, show_labels):
        if result is None or x > result.n_components or y > result.n_components:
            return pn.pane.Markdown("")
        df = self._frame(result)
        xcol, ycol = f"PC{x}", f"PC{y}"
        ratio = result.explained_variance_ratio
        points = []
        for name, group in df.groupby("set", sort=False):
            points.append(
                hv.Scatter(
                    group,
                    kdims=[xcol],
                    vdims=[ycol, "sample", "sample_sets"],
                    label=name,
                ).opts(color=self._colors.get(name, "grey"), size=12, tools=["hover"])
            )
        plot = hv.Overlay(points)
        if show_labels:
            plot = plot * hv.Labels(df, [xcol, ycol], "sample").opts(
                text_font_size="8pt", xoffset=0.02, text_align="left"
            )
        return plot.opts(
            hv.opts.Overlay(
                xlabel=f"{xcol} ({ratio[x - 1]:.1%})",
                ylabel=f"{ycol} ({ratio[y - 1]:.1%})",
                legend_position="right",
                height=450,
                responsive=True,
                title="Samples on the principal components",
            )
        )

    def _scree(self, result):
        if result is None:
            return pn.pane.Markdown("")
        df = result.variance(MAX_COMPONENTS_TO_CHOOSE)
        return hv.Bars(df, "component", "explained_variance_ratio").opts(
            ylabel="Explained variance ratio",
            title=f"Explained variance, first {len(df)} components",
            height=250,
            responsive=True,
            color="#0072B2",
        )

    def _table(self, result):
        if result is None:
            return pn.pane.Markdown("")
        df = result.variance(MAX_COMPONENTS_TO_CHOOSE)
        return pn.widgets.Tabulator(
            df, disabled=True, show_index=False, layout="fit_data_table"
        )

    def _scores_tsv(self):
        if self.result is None:
            return StringIO("")
        return StringIO(
            self.result.scores(sample_sets=self._sets).to_csv(sep="\t", index=False)
        )

    def __panel__(self):
        window = pn.widgets.IntInput.from_param(
            self.param.window_size, name="Window size (bp)", width=160
        )
        status = pn.bind(
            self._status,
            self.param.result,
            self.param.ran_for,
            self.state.param.coord,
            self.param.window_size,
        )
        scatter = pn.bind(
            self._scatter,
            self.param.result,
            self.param.x_component,
            self.param.y_component,
            self.param.show_labels,
        )
        return pn.Column(
            "# PCA",
            pn.pane.Markdown(
                "Principal components of the coverage with the samples as the "
                "features: the mean coverage of every sample in windows inside "
                "the selected contigs, divided by the sample's mean coverage, "
                "centred. Each sample gets a loading on every component."
            ),
            pn.Row(window, pn.Column(pn.Spacer(height=20), self._button)),
            status,
            pn.Row(
                pn.widgets.Select.from_param(self.param.x_component, name="x"),
                pn.widgets.Select.from_param(self.param.y_component, name="y"),
                pn.widgets.Checkbox.from_param(
                    self.param.show_labels, name="Sample names"
                ),
                self._download,
            ),
            scatter,
            pn.bind(self._scree, self.param.result),
            pn.bind(self._table, self.param.result),
        )
