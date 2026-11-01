import math
from pathlib import Path

import click

from pbview import config
from pbview.logging import cli_logger as logger
from pbview.logging import log_level
from pbview.model.coordinates import Coordinates
from pbview.model.datastore import DataStore

from ._common import (
    memory_limit_option,
    path_argument,
    progress_option,
    set_threads,
    threads_option,
    track_name_option,
    workers_option,
)


def pca_paths(prefix: str) -> tuple[Path, Path]:
    """Files written for the output prefix: scores and variance."""
    return Path(f"{prefix}.scores.tsv"), Path(f"{prefix}.variance.tsv")


@click.command("pca")
@path_argument(exists=True, dir_okay=True, nargs=1)
@workers_option(default=1)
@threads_option(default=1)
@memory_limit_option()
@track_name_option()
@log_level()
@progress_option()
@click.option(
    "--window-size",
    type=click.IntRange(min=1),
    default=config.PCA_WINDOW_SIZE,
    show_default=True,
    help=(
        "Positions per window. The data are the mean coverage of every sample "
        "in non-overlapping windows inside the contigs; contigs shorter than "
        "this have none. Env: PBVIEW_PCA_WINDOW_SIZE"
    ),
)
@click.option(
    "--n-components",
    type=click.IntRange(min=1),
    default=None,
    help="Number of components to write (default: all, one per sample)",
)
@click.option(
    "--min-contig-length",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Only use contigs at least this long",
)
@click.option(
    "--max-contig-length",
    type=click.IntRange(min=0),
    default=None,
    help="Only use contigs at most this long (default: no limit)",
)
@click.option(
    "--sample-set",
    "sample_sets",
    multiple=True,
    help="Only use the samples of this sample set (repeatable; default: all)",
)
@click.option(
    "-o",
    "--output",
    required=True,
    help=(
        "Output prefix: writes PREFIX.scores.tsv (the loading of each sample "
        "on each component) and PREFIX.variance.tsv (eigenvalue and explained "
        "variance of each component)"
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing output files",
)
@set_threads
def pca(
    path: Path | str,
    track_name: str,
    progress: bool,
    window_size: int,
    n_components: int | None,
    min_contig_length: int,
    max_contig_length: int | None,
    sample_sets: tuple[str, ...],
    output: str,
    force: bool,
) -> None:
    """Principal component analysis of the coverage of the samples.

    The samples are the features: each sample is divided by its mean
    coverage (so depth is not the first component), centred, and gets a
    loading on every component. One pass over the data, out of core.
    """
    scores_file, variance_file = pca_paths(output)
    existing = [str(f) for f in (scores_file, variance_file) if f.exists()]
    if existing and not force:
        raise click.ClickException(
            f"{', '.join(existing)} exists; use --force to overwrite"
        )
    ds = DataStore(path=path)
    if track_name not in ds.tracks:
        raise click.ClickException(
            f"no track {track_name!r} (available: {sorted(ds.tracks)})"
        )
    track = ds.tracks[track_name]
    coord = Coordinates().with_contigs(
        ds.genome.contigs_in_length_range(
            min_contig_length,
            math.inf if max_contig_length is None else max_contig_length,
        )
    )
    if sample_sets:
        coord = coord.with_sample_sets(sample_sets)
    try:
        resolved = coord.resolve_for(track, ds.genome)
        if progress:
            logger.info("Computing PCA of %d samples", len(resolved.labels))
        result = track.pca(resolved, window_size)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    membership = track.group_membership()
    result.scores(n_components, membership).to_csv(
        scores_file, sep="\t", index=False, float_format="%.8g"
    )
    result.variance(n_components).to_csv(
        variance_file, sep="\t", index=False, float_format="%.8g"
    )
    ratio = result.explained_variance_ratio
    click.echo(
        f"PCA of {len(result.samples)} samples in {result.n_windows} windows of "
        f"{window_size} bp; PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%} of the variance"
    )
    click.echo(f"Wrote {scores_file} and {variance_file}")
