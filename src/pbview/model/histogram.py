"""Histogram statistical functions"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-09-17"

import numpy as np

from pbview import config
from pbview.model.coordinates import Coordinates
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


def fold_hist(hist, n_bins: int) -> np.ndarray:
    """Histogram of values clipped to `n_bins - 1`, from an unclipped one.

    Bins `n_bins - 1` and above are summed into the last bin; a shorter
    histogram is padded with zeros. This equals re-binning the underlying
    values after `np.minimum(values, n_bins - 1)`, without another pass.
    """
    hist = np.asarray(hist)
    out = np.zeros(n_bins, dtype=hist.dtype)
    m = min(n_bins, len(hist))
    out[:m] = hist[:m]
    out[n_bins - 1] += hist[n_bins:].sum()
    return out


def sample_set_hists(
    datastore: "DataStore", track_name: str, max_positions: int
) -> dict[str, np.ndarray]:
    """`Track.sum_hists` for every sample set of a track, read in one pass.

    The result is cached on the track, so asking again for the same
    `max_positions` does not read the data again.
    """
    track = datastore.tracks[track_name]
    sets = tuple(track.groups("sample_set"))
    n_samples = tuple(
        len(
            Coordinates()
            .with_sample_sets([s])
            .resolve_for(track, datastore.genome)
            .labels
        )
        for s in sets
    )
    return track.sum_hists(sets, n_samples, max_positions)


def compute_threshold_defaults(
    datastore: "DataStore",
    active_track: str,
    *,
    max_bin_q: float = 0.995,
    max_bin_headroom: float = 1.2,
    max_positions: int | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Compute threshold defaults for each sample set in a track.

    Return (lower, upper, max_bin) for each sample set in a track,
    where lower=0.6*mean, upper=median+2*std, and max_bin=max_bin_q
    quantile * max_bin_headroom. Mean and median values are computed
    across the entire genome and may therefore not be representative
    of the actual coverage distribution in a given region.

    The genome is read once for all sample sets (`Track.sum_hists`); the
    histogram clipped at `max_bin` that gives `upper` is derived from it.

    With more than `max_positions` positions (default
    `config.MAX_POSITIONS`; 0 uses all) the statistics are estimated
    from evenly spaced positions, which makes them estimates.
    """
    if max_positions is None:
        max_positions = config.MAX_POSITIONS
    result = {}
    for s, h in sample_set_hists(datastore, active_track, max_positions).items():
        stats = hist_stats(h)
        lo = int(0.6 * stats["mean"])
        mb = int(hist_quantile(h, max_bin_q) * max_bin_headroom)
        counts = fold_hist(h, mb)  # the histogram clipped at mb - 1
        stats = hist_stats(counts)
        hi = int(stats["median"] + 2 * stats["std"])
        result[f"lower_coverage_{s}"] = lo
        result[f"upper_coverage_{s}"] = hi
        result[f"maxbins_{s}"] = mb
    return result
