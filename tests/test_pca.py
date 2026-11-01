import dask.array as da
import numpy as np
import pytest

from pbview.model.coordinates import PositionIntervals
from pbview.model.pca import (
    Moments,
    PcaResult,
    pca_from_moments,
    window_moments,
    window_starts,
)

SAMPLES = ("a", "b", "c", "d")


def reference_pca(windows):
    """PCA of the samples (columns) of a window matrix, with numpy only.

    Scale every sample by its mean, centre it, and take the SVD.
    """
    scaled = windows / windows.mean(axis=0)
    scaled = scaled - scaled.mean(axis=0)
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    eigenvalues = singular**2 / (len(windows) - 1)
    return eigenvalues, vt.T


def window_matrix(values, starts, size):
    return np.array([values[s : s + size].mean(axis=0) for s in starts])


@pytest.fixture
def values():
    rng = np.random.default_rng(3)
    base = rng.gamma(4.0, 5.0, size=(240, 1))  # shared coverage profile
    depth = np.array([1.0, 2.0, 0.5, 3.0])
    noise = rng.normal(1.0, 0.15, size=(240, 4))
    return np.round(base * depth * noise).astype(np.int32)


# ---- window_starts


def test_windows_start_at_the_contig_start_and_never_cross_it():
    offsets = [0, 10, 18, 24]  # contigs of 10, 8 and 6
    everything = PositionIntervals(((0, 24),))
    assert window_starts(offsets, everything, 4).tolist() == [0, 4, 10, 14, 18]
    assert window_starts(offsets, everything, 3).tolist() == [0, 3, 6, 10, 13, 18, 21]


def test_contigs_shorter_than_the_window_have_no_windows():
    offsets = [0, 10, 18, 24]
    everything = PositionIntervals(((0, 24),))
    assert window_starts(offsets, everything, 8).tolist() == [0, 10]
    assert window_starts(offsets, everything, 11).tolist() == []


def test_only_selected_contigs_have_windows():
    offsets = [0, 10, 18, 24]
    chr2 = PositionIntervals(((10, 18),))
    assert window_starts(offsets, chr2, 4).tolist() == [10, 14]
    two = PositionIntervals(((0, 10), (18, 24)))
    assert window_starts(offsets, two, 5).tolist() == [0, 5, 18]
    assert window_starts(offsets, PositionIntervals(()), 4).tolist() == []


def test_adjacent_selected_contigs_are_merged_but_keep_their_borders():
    offsets = [0, 10, 18, 24]
    merged = PositionIntervals(((0, 18),))  # chr1 and chr2 as one interval
    assert window_starts(offsets, merged, 4).tolist() == [0, 4, 10, 14]


def test_window_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        window_starts([0, 10], PositionIntervals(((0, 10),)), 0)


# ---- window_moments


@pytest.mark.parametrize("chunks", [240, 100, 64, 7, 3, (1, 99, 140)])
def test_moments_do_not_depend_on_the_chunking(values, chunks):
    size = 10
    offsets = np.array([0, 100, 240])
    starts = window_starts(offsets, PositionIntervals(((0, 240),)), size)
    expected = window_matrix(values, starts, size)
    moments = window_moments(da.from_array(values, chunks=(chunks, 4)), starts, size)
    assert moments.n_windows == len(starts) == 24
    np.testing.assert_allclose(moments.s1, expected.sum(axis=0))
    np.testing.assert_allclose(moments.s2, expected.T @ expected)


def test_moments_of_a_window_that_crosses_a_chunk_border(values):
    starts = np.array([8])  # window 8..18, chunk border at 10
    moments = window_moments(da.from_array(values, chunks=(10, 4)), starts, 10)
    expected = values[8:18].mean(axis=0)
    np.testing.assert_allclose(moments.s1, expected)


def test_moments_only_read_chunks_that_hold_windows(values):
    reads = []

    def read(block):
        if block.shape[0]:  # dask calls it on an empty block to find the dtype
            reads.append(block.shape[0])
        return block

    array = da.from_array(values, chunks=(60, 4)).map_blocks(read, dtype=values.dtype)
    starts = np.array([130])  # chunk 2 (120..180), window 130..140
    window_moments(array, starts, 10)
    assert reads == [60]  # one chunk read, once


def test_every_chunk_is_read_once_also_for_windows_across_its_borders(values):
    reads = []

    def read(block):
        if block.shape[0]:
            reads.append(block.shape[0])
        return block

    array = da.from_array(values, chunks=(60, 4)).map_blocks(read, dtype=values.dtype)
    # windows 55..65 and 115..125 cross the borders at 60 and 120; the last
    # window, 170..180, ends at the border 180
    starts = np.array([55, 115, 170])
    moments = window_moments(array, starts, 10)
    assert sorted(reads) == [60, 60, 60]  # chunks 0 to 2, once each, not chunk 3
    expected = np.array([values[s : s + 10].mean(axis=0) for s in starts])
    np.testing.assert_allclose(moments.s1, expected.sum(axis=0))
    np.testing.assert_allclose(moments.s2, expected.T @ expected)


def test_windows_longer_than_the_chunks(values):
    starts = np.array([5, 40])  # each spans several chunks of 7
    moments = window_moments(da.from_array(values, chunks=(7, 4)), starts, 30)
    expected = np.array([values[s : s + 30].mean(axis=0) for s in starts])
    assert moments.n_windows == 2
    np.testing.assert_allclose(moments.s1, expected.sum(axis=0))
    np.testing.assert_allclose(moments.s2, expected.T @ expected)


def test_no_windows_gives_empty_moments(values):
    moments = window_moments(da.from_array(values, chunks=(60, 4)), np.array([]), 10)
    assert moments.n_windows == 0
    assert moments.s2.shape == (4, 4)


# ---- pca_from_moments


@pytest.mark.parametrize("chunks", [240, 25])
def test_pca_equals_numpy_pca_of_the_window_matrix(values, chunks):
    size = 6
    starts = window_starts([0, 100, 240], PositionIntervals(((0, 240),)), size)
    windows = window_matrix(values, starts, size)
    result = pca_from_moments(
        window_moments(da.from_array(values, chunks=(chunks, 4)), starts, size),
        SAMPLES,
        size,
    )
    eigenvalues, vectors = reference_pca(windows)
    np.testing.assert_allclose(result.eigenvalues, eigenvalues, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.abs(result.components), np.abs(vectors), atol=1e-8)
    np.testing.assert_allclose(
        np.abs(result.loadings), np.abs(vectors * np.sqrt(eigenvalues)), atol=1e-8
    )
    np.testing.assert_allclose(
        result.explained_variance_ratio, eigenvalues / eigenvalues.sum()
    )
    np.testing.assert_allclose(result.mean_coverage, windows.mean(axis=0))
    assert result.n_windows == len(starts)
    assert result.samples == SAMPLES


def test_depth_scaling_removes_the_depth_component():
    # two samples, one has exactly twice the coverage of the other
    profile = np.array([5.0, 9.0, 3.0, 12.0, 7.0, 4.0])
    windows = np.column_stack([profile, 2 * profile, profile * 3])
    moments = Moments(len(windows), windows.sum(0), windows.T @ windows)
    result = pca_from_moments(moments, ("a", "b", "c"), 1)
    assert result.explained_variance_ratio[0] == pytest.approx(1.0)
    # all samples load equally on the one component
    assert np.ptp(np.abs(result.components[:, 0])) < 1e-12


def test_component_signs_are_fixed_and_components_are_orthonormal(values):
    starts = window_starts([0, 240], PositionIntervals(((0, 240),)), 6)
    moments = window_moments(da.from_array(values, chunks=(50, 4)), starts, 6)
    result = pca_from_moments(moments, SAMPLES, 6)
    np.testing.assert_allclose(
        result.components.T @ result.components, np.eye(4), atol=1e-10
    )
    largest = np.abs(result.components).argmax(axis=0)
    assert (result.components[largest, np.arange(4)] > 0).all()
    assert (np.diff(result.eigenvalues) <= 1e-12).all()  # largest first


def test_pca_tables():
    windows = np.arange(1.0, 25).reshape(6, 4) ** 1.5
    result = pca_from_moments(
        Moments(6, windows.sum(0), windows.T @ windows), SAMPLES, 5
    )
    scores = result.scores(2, sample_sets={s: ("pop1", "pop2") for s in SAMPLES})
    assert list(scores.columns) == ["sample", "sample_sets", "PC1", "PC2"]
    assert scores["sample"].tolist() == list(SAMPLES)
    assert scores["sample_sets"].unique().tolist() == ["pop1,pop2"]
    np.testing.assert_allclose(scores["PC1"], result.loadings[:, 0])
    assert list(result.scores().columns) == ["sample", "PC1", "PC2", "PC3", "PC4"]
    variance = result.variance(3)
    assert variance["component"].tolist() == ["PC1", "PC2", "PC3"]
    np.testing.assert_allclose(variance["eigenvalue"], result.eigenvalues[:3])
    assert isinstance(result, PcaResult)


def test_pca_needs_two_samples():
    with pytest.raises(ValueError, match="at least two samples"):
        pca_from_moments(Moments(5, np.ones(1), np.ones((1, 1))), ("a",), 5)


def test_pca_needs_two_windows():
    with pytest.raises(ValueError, match="at least two windows of 5 bp, got 1"):
        pca_from_moments(Moments(1, np.ones(2), np.ones((2, 2))), ("a", "b"), 5)


def test_pca_names_the_samples_without_coverage():
    windows = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    moments = Moments(3, windows.sum(0), windows.T @ windows)
    with pytest.raises(ValueError, match=r"without coverage.*\['b'\]"):
        pca_from_moments(moments, ("a", "b"), 5)
