"""Histogram statistical functions"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import numpy as np

from pbview.model.datastore import DataStore


def hist_stats(hist, values=None) -> dict:
    """Compute min, mean, median, max, stdev from a histogram.

    Args:
        hist: 1-D array of counts; hist[i] = number of observations
            with value values[i].
        values: 1-D array of the value for each bin. If None, uses
                np.arange(len(hist)) (appropriate for bincount output
                on integer data).

    Returns:
        dict with min, mean, median, max, std.
    """
    hist = np.asarray(hist)
    if values is None:
        values = np.arange(len(hist))

    n = hist.sum()
    if n == 0:
        return dict(min=np.nan, mean=np.nan, median=np.nan, max=np.nan, std=np.nan)

    nz = np.flatnonzero(hist)
    lo, hi = nz[0], nz[-1]

    mean = (values * hist).sum() / n
    var = ((values - mean) ** 2 * hist).sum() / n

    cum = np.cumsum(hist)
    median_idx = np.searchsorted(cum, n / 2)
    median = values[median_idx]

    return dict(
        min=values[lo],
        mean=mean,
        median=median,
        max=values[hi],
        std=np.sqrt(var),
    )


def hist_boxplot_stats(hist, values=None) -> dict:
    hist = np.asarray(hist)
    if values is None:
        values = np.arange(len(hist))
    n = hist.sum()
    if n == 0:
        return {k: np.nan for k in ["min", "q1", "median", "q3", "max"]}

    cum = np.cumsum(hist)

    def q(p):
        return values[np.searchsorted(cum, p * n)]

    nz = np.flatnonzero(hist)
    return {
        "min": values[nz[0]],
        "q1": q(0.25),
        "median": q(0.50),
        "q3": q(0.75),
        "max": values[nz[-1]],
    }


def hist_quantile(hist, q, values=None):
    """Quantile(s) from histogram counts.

    Args:
        hist: 1-D counts.
        q: scalar or array of quantiles in [0, 1].
        values: bin values; defaults to np.arange(len(hist)).

    Returns:
        Same shape as q.
    """
    hist = np.asarray(hist)
    if values is None:
        values = np.arange(len(hist))
    n = hist.sum()
    if n == 0:
        return (
            np.nan
            if np.isscalar(q)
            else np.full_like(np.asarray(q, dtype=float), np.nan)
        )

    cum = np.cumsum(hist)
    targets = np.atleast_1d(q) * n
    idx = np.searchsorted(cum, targets, side="left")
    idx = np.clip(idx, 0, len(values) - 1)
    result = values[idx]
    return result[0] if np.isscalar(q) else result


def compute_threshold_defaults(
    datastore: "DataStore",
    active_track: str,
    *,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    max_bin_q: float = 0.995,
    max_bin_headroom: float = 1.2,
) -> dict[str, tuple[int, int, int]]:
    """Compute threshold defaults for each sample set in a track.


    Return (lower, upper, max_bin) for each sample set in a track"""
    result = {}
    track = datastore.tracks[active_track]
    for s in datastore.base_coord.sample_set_names:
        coord = datastore.base_coord.with_sample_sets([s])
        h = track._pre.sum_hist(s)
        stats = hist_stats(h)
        lo = int(0.6 * stats["mean"])
        mb = int(hist_quantile(h, max_bin_q) * max_bin_headroom)
        bins = np.arange(mb)
        counts, bins = track.coverage_hist(bins=bins, coord=coord)
        stats = hist_stats(counts, bins)
        hi = int(stats["median"] + 2 * stats["std"])
        result[f"lower_coverage_{s}"] = lo
        result[f"upper_coverage_{s}"] = hi
        result[f"maxbins_{s}"] = mb
    return result
