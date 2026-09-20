"""
Plot helpers
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-20"

import holoviews as hv
import numpy as np

from ..model.histogram import hist_boxplot_stats


def hist_boxplot(
    hists: dict[str, np.ndarray],
    values: dict[str, np.ndarray] | None = None,
    color_map: dict[str, str] | None = None,
    **opts,
) -> hv.Overlay:
    """
    Boxplot per sample set, built from precomputed histograms.

    Returns:
        Holoviews Overlay of Rectangles + Segments + Scatter, one box
        per sample set.
    """
    names = list(hists)
    if values is None:
        values = {k: np.arange(len(v)) for k, v in hists.items()}
    if color_map is None:
        import colorcet as cc

        palette = cc.glasbey_category10
        color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    box_w = 0.6  # box width
    box_data = []  # (x0, y0, x1, y1, color)
    median_data = []  # (x0, median, x1, median, color)
    whisker_data = []  # (x, y0, x, y1, color)
    cap_data = []  # (x0, y, x1, y, color)
    hover_data = []  # for tooltip layer

    for i, name in enumerate(names):
        s = hist_boxplot_stats(hists[name], values[name])
        color = color_map[name]
        x = i

        box_data.append((x - box_w / 2, s["q1"], x + box_w / 2, s["q3"], color))
        median_data.append(
            (x - box_w / 2, s["median"], x + box_w / 2, s["median"], color)
        )
        whisker_data.append((x, s["min"], x, s["q1"], color))
        whisker_data.append((x, s["q3"], x, s["max"], color))
        cap_data.append((x - box_w / 4, s["min"], x + box_w / 4, s["min"], color))
        cap_data.append((x - box_w / 4, s["max"], x + box_w / 4, s["max"], color))
        hover_data.append(
            (x, s["median"], name, s["min"], s["q1"], s["median"], s["q3"], s["max"])
        )

    boxes = hv.Rectangles(
        box_data, kdims=["x0", "y0", "x1", "y1"], vdims=["color"]
    ).opts(color="color", line_color="black", fill_alpha=0.5)

    medians = hv.Segments(
        median_data, kdims=["x0", "y0", "x1", "y1"], vdims=["color"]
    ).opts(color="black", line_width=2)

    whiskers = hv.Segments(
        whisker_data, kdims=["x0", "y0", "x1", "y1"], vdims=["color"]
    ).opts(color="color", line_width=1)

    caps = hv.Segments(cap_data, kdims=["x0", "y0", "x1", "y1"], vdims=["color"]).opts(
        color="color", line_width=1
    )

    # invisible scatter for hover tooltips
    hover = hv.Scatter(
        hover_data,
        kdims=["x"],
        vdims=["median", "sample_set", "min", "q1", "med", "q3", "max"],
    ).opts(alpha=0, tools=["hover"], size=20)

    return (boxes * medians * whiskers * caps * hover).opts(
        xticks=list(enumerate(names)),
        xlabel="Sample set",
        ylabel="Coverage",
        show_legend=False,
        **opts,
    )
