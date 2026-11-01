"""Out-of-core PCA of coverage, with the samples as the features.

The data matrix has one row per window and one column per sample: the
mean coverage of the sample in a window of `window_size` bp that lies
inside one contig. The PCA is that of scikit-learn: every sample is a
feature, centred over the windows, and the components live in sample
space, so each sample gets a coordinate on every component (its loading).

Coverage differs between samples with the sequencing depth, which would
make the first component a depth component. Every sample is therefore
divided by its mean coverage first. That is a diagonal scaling of the
covariance matrix, `C_ij = cov_ij / (mean_i mean_j)`, so a single pass
over the data is enough: for the windows it accumulates their number, the
sum of the window means and the sum of their outer products. Only
n_samples x n_samples numbers are kept, however large the genome is.
"""

from __future__ import annotations

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__date__ = "2026-11-01"

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pbview.model.coordinates import PositionIntervals


@dataclass(frozen=True)
class Moments:
    """Sums over windows of the window means.

    Attributes:
        n_windows: number of windows
        s1: sum of the window means, shape (n_samples,)
        s2: sum of the outer products of the window means,
            shape (n_samples, n_samples)
    """

    n_windows: int
    s1: np.ndarray
    s2: np.ndarray


@dataclass(frozen=True, eq=False)
class PcaResult:
    """PCA of the samples.

    Attributes:
        samples: the samples, in the order of the rows of `components`
        eigenvalues: variance of each component, largest first
        explained_variance_ratio: `eigenvalues` as a fraction of their sum
        components: unit eigenvectors, shape (n_samples, n_components);
            the sign is fixed so that the largest entry of each is positive
        loadings: `components * sqrt(eigenvalues)`, the coordinates the
            samples are plotted at
        mean_coverage: mean coverage of each sample over the windows
        n_windows: number of windows
        window_size: window size in bp
    """

    samples: tuple[str, ...]
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    loadings: np.ndarray
    mean_coverage: np.ndarray
    n_windows: int
    window_size: int

    @property
    def n_components(self) -> int:
        return len(self.eigenvalues)

    def scores(
        self,
        n_components: int | None = None,
        sample_sets: dict[str, Sequence[str]] | None = None,
    ) -> pd.DataFrame:
        """Loadings of the samples as a table: sample, [sample_sets], PC1, ...

        Args:
            n_components: number of components, default all
            sample_sets: sample sets of each sample, added comma separated
        """
        k = self.n_components if n_components is None else n_components
        k = min(k, self.n_components)
        df = pd.DataFrame({"sample": list(self.samples)})
        if sample_sets is not None:
            df["sample_sets"] = [",".join(sample_sets[s]) for s in self.samples]
        for i in range(k):
            df[f"PC{i + 1}"] = self.loadings[:, i]
        return df

    def variance(self, n_components: int | None = None) -> pd.DataFrame:
        """Eigenvalue and explained variance ratio of each component."""
        k = self.n_components if n_components is None else n_components
        k = min(k, self.n_components)
        return pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(k)],
                "eigenvalue": self.eigenvalues[:k],
                "explained_variance_ratio": self.explained_variance_ratio[:k],
            }
        )


def window_starts(
    offsets: Iterable[int], intervals: PositionIntervals, window_size: int
) -> np.ndarray:
    """Start of every full window of every selected contig.

    Windows are `window_size` positions long, do not overlap, start at the
    start of the contig and never cross a contig border; the last
    `length % window_size` positions of a contig are not used, and a
    contig shorter than the window has no window.

    Args:
        offsets: contig borders: contig i is `offsets[i]:offsets[i + 1]`
        intervals: selected positions, a union of whole contigs (adjacent
            contigs are merged, so the borders come from `offsets`)
        window_size: window length in positions

    Returns:
        Start positions in increasing order.
    """
    if window_size < 1:
        raise ValueError(f"window size must be at least 1, got {window_size}")
    offsets = np.asarray(offsets, dtype=np.int64)
    first, last = offsets[:-1], offsets[1:]
    ivs = np.asarray(intervals.intervals, dtype=np.int64).reshape(-1, 2)
    if len(ivs) == 0:
        return np.empty(0, dtype=np.int64)
    ivs = ivs[np.argsort(ivs[:, 0])]
    i = np.searchsorted(ivs[:, 0], first, side="right") - 1
    inside = (i >= 0) & (last <= ivs[np.maximum(i, 0), 1])
    first, last = first[inside], last[inside]
    counts = (last - first) // window_size
    total = int(counts.sum())
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(first, counts) + within * window_size


def _chunk_moments(block: np.ndarray, starts: np.ndarray, window_size: int) -> Moments:
    """Moments of the windows that start at `starts` in `block` (rows are positions)."""
    dtype = np.int64 if np.issubdtype(block.dtype, np.integer) else np.float64
    sums = np.empty((len(starts), block.shape[1]), dtype=dtype)
    # Windows that follow each other (the rule inside a contig) are summed
    # together as one reshaped slice, which is ten times faster than
    # np.add.reduceat on a block of a million positions.
    follows = np.diff(starts, prepend=starts[0] - window_size - 1) == window_size
    first = np.flatnonzero(~follows)
    last = np.append(first[1:], len(starts))
    for a, b in zip(first, last):
        start = int(starts[a])
        run = block[start : start + (b - a) * window_size]
        sums[a:b] = run.reshape(b - a, window_size, -1).sum(axis=1, dtype=dtype)
    means = sums / window_size
    return Moments(len(starts), means.sum(axis=0), means.T @ means)


def _chunk_partials(
    block: np.ndarray,
    offset: int,
    inside: np.ndarray,
    crossing: np.ndarray,
    crossing_ids: np.ndarray,
    window_size: int,
) -> tuple[Moments | None, dict[int, np.ndarray]]:
    """What one chunk contributes.

    The moments of the windows that lie inside the chunk (`inside`, their
    starts), and for each window that crosses a border of the chunk
    (`crossing`, their starts, with their numbers `crossing_ids`) the sum of
    the rows of the window that are in this chunk.
    """
    moments = None
    if len(inside):
        moments = _chunk_moments(block, inside - offset, window_size)
    dtype = np.int64 if np.issubdtype(block.dtype, np.integer) else np.float64
    partial = {}
    for start, number in zip(crossing, crossing_ids):
        lo = max(int(start) - offset, 0)
        hi = min(int(start) + window_size - offset, block.shape[0])
        partial[int(number)] = block[lo:hi].sum(axis=0, dtype=dtype)
    return moments, partial


def window_moments(values: da.Array, starts: np.ndarray, window_size: int) -> Moments:
    """Moments of the window means of a (position, sample) array, out of core.

    Every chunk that holds part of a window is read once, and no other. A
    window inside one chunk is summed by the task of that chunk. A window
    that crosses a chunk border gets a partial sum from each chunk it
    touches, and the partial sums are added up here, so the result does not
    depend on the chunking, nor does a chunk have to be read for its
    neighbour.

    Args:
        values: coverage, shape (position, sample)
        starts: start positions of the windows, increasing (`window_starts`)
        window_size: window length in positions
    """
    n_samples = values.shape[1]
    s1 = np.zeros(n_samples)
    s2 = np.zeros((n_samples, n_samples))
    n_windows = 0
    if len(starts):
        values = values.rechunk({1: -1})
        bounds = np.concatenate([[0], np.cumsum(values.chunks[0])])
        first = np.searchsorted(bounds, starts, side="right") - 1
        last = np.searchsorted(bounds, starts + window_size - 1, side="right") - 1
        crossing_ids = np.flatnonzero(first != last)
        cross_first, cross_last = first[crossing_ids], last[crossing_ids]
        tasks = []
        for i in range(len(bounds) - 1):
            lo, hi = np.searchsorted(first, [i, i + 1])
            inside = np.arange(lo, hi)
            inside = inside[last[inside] == i]  # not the ones that cross
            # crossing windows are in increasing order, so those that touch chunk i
            # are a contiguous range
            c_lo = np.searchsorted(cross_last, i, side="left")
            c_hi = np.searchsorted(cross_first, i, side="right")
            ids = crossing_ids[c_lo:c_hi]
            if len(inside) == 0 and len(ids) == 0:
                continue
            tasks.append(
                dask.delayed(_chunk_partials)(
                    values.blocks[i, 0],
                    int(bounds[i]),
                    starts[inside],
                    starts[ids],
                    ids,
                    window_size,
                )
            )
        sums: dict[int, np.ndarray] = {}
        for moments, partial in dask.compute(*tasks):
            if moments is not None:
                n_windows += moments.n_windows
                s1 += moments.s1
                s2 += moments.s2
            for number, vector in partial.items():
                sums[number] = sums.get(number, 0) + vector
        if sums:
            means = np.array([v / window_size for v in sums.values()], dtype=np.float64)
            n_windows += len(means)
            s1 += means.sum(axis=0)
            s2 += means.T @ means
    return Moments(n_windows, s1, s2)


def pca_from_moments(
    moments: Moments, samples: Sequence[str], window_size: int
) -> PcaResult:
    """PCA of the samples from the moments of the window means.

    Raises:
        ValueError: with fewer than two samples or two windows, or if a
            sample has no coverage, which makes scaling by its depth impossible
    """
    samples = tuple(samples)
    n, windows = len(samples), moments.n_windows
    if n < 2:
        raise ValueError(f"PCA needs at least two samples, got {n}")
    if windows < 2:
        raise ValueError(
            f"PCA needs at least two windows of {window_size} bp, got {windows}; "
            "use a smaller window size or select longer contigs"
        )
    mean = moments.s1 / windows
    empty = [s for s, m in zip(samples, mean) if m <= 0]
    if empty:
        raise ValueError(
            f"samples without coverage cannot be scaled by their depth: {empty}"
        )
    cov = (moments.s2 - windows * np.outer(mean, mean)) / (windows - 1)
    scaled = cov / np.outer(mean, mean)
    scaled = (scaled + scaled.T) / 2
    eigenvalues, vectors = np.linalg.eigh(scaled)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    vectors = vectors[:, order]
    largest = np.argmax(np.abs(vectors), axis=0)
    vectors = vectors * np.where(vectors[largest, np.arange(n)] < 0, -1.0, 1.0)
    total = eigenvalues.sum()
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return PcaResult(
        samples=samples,
        eigenvalues=eigenvalues,
        explained_variance_ratio=ratio,
        components=vectors,
        loadings=vectors * np.sqrt(eigenvalues),
        mean_coverage=mean,
        n_windows=windows,
        window_size=window_size,
    )
