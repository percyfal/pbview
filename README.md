# pbview

## About

pbview is an application built on
[pbzarr](https://github.com/pbzarr/pbzarr) that provides a CLI and
visualiser to analyze and view genomic data at a base-pair resolution.
Originally developed to deal with sequencing coverage data, current
functionality includes summarizing data across sample sets and
selecting coverage thresholds for generating accessibility masks.

## Status

pbview is under active development and does not yet have an official
release. The final data structure implementation depends heavily on
the pbzarr specification, which itself is in active development.
Therefore, data implementation details may change with short notice,
and CLI/API interfaces may change to provide new functionality.

## PBZ datastore format and extensions

pbview uses the pbzarr [PBZ design
specification](https://github.com/pbzarr/pbzarr/blob/main/docs/DESIGN.md)
for organizing data, and adds additional functionality on top. A `pbz`
archive is a collection of `tracks`, each of which stores per-base
numeric data on a genome reference coordinate system. The following
sections briefly describe `pbview` data conventions and how they
differ from the main `pbz` specification. For more details, see the
[design document](docs/design.md).

### Global genome coordinate system

A main feature of pbview is to allow for summaries of tracks in which
the columns of a track are grouped into smaller analysis units. The
immediate consequence is that track genome coordinates must be shared
among tracks and their summaries. Therefore, pbview makes the
assumption that the reference coordinate system is shared among
**all** tracks, in contrast to the current PBZ specification (v0.4)
where each track owns its own genome. Spatial selections are global
and should have identical effects on all tracks (the genome is read
from one track and is not currently checked against the others). This
convention separates global spatial selections from column selections that apply
to individual tracks (see following paragraph).

> [!NOTE]
> Although the current PBZ specification lets tracks own their own
> genomes, this may change to a global coordinate system sitting at
> the root in future versions, in line with the pbview data model.

### Arbitrary column dimensions

Whereas a pbzarr `pbz` track contains `values` with dimensions
`(position, sample)`, pbview allows arbitrary column dimension names,
`<column_dim>` (the `position` dimension is shared among all tracks).
Commonly, `<column_dim>=sample`. The data model and the selection
layer treat all column dimension names alike, but the current app
assumes that the base track has `sample` columns.

### Groupings

It is often of interest to group measurement units to calculate
summaries, such as sums or mean values. A track can therefore hold a
grouping which is a boolean `(column, group)` membership matrix. For
example, a `(sample, sample_set)` grouping on a track with `sample`
columns would map samples to sample sets (e.g., populations). pbview
can utilize this information to generate summary tracks with summaries
across the groupings, where values have dimension `(position, group)`.

Every track also has an implicit *identity* grouping with the single
set `ALL`, which is computed and never stored. Stored groupings, such
as the one created from `--sampleset-info`, always include an `ALL` set
that contains every column, so a summary track always has an `ALL`
column.

### Extension: track metadata

Whereas the previous conventions are all compatible with the current
`pbz` data model, pbview adds metadata in two places. Groupings are
stored in `<track>/_meta/column_groupings`. Summary tracks record
their origin in attributes (`perbase:source_track`,
`perbase:provenance`, `perbase:source_grouping_hash` and, for
missingness, `perbase:missing_cutoff`), so they are found by their
attributes and never by name. An example for sequencing coverage is
shown below:

```text
example.pbz/
├── zarr.json                  # root marker + consolidated metadata
├── depth/
│   ├── _meta
│   │   └── column_groupings   # xr.Dataset, (sample, sample_set) grouping
│   ├── zarr.json              # perbase: interpretation attributes
│   ├── values                 # (ΣL, n_columns), (position, sample)
│   ├── offsets                # int64[k + 1], (contig_boundary,)
│   ├── contigs                # vlen-utf8[k], (contig,)
│   └── sample                 # vlen-utf8[n_columns], (sample,)
└── depth_sum/
    ├── zarr.json              # perbase:source_track, provenance, ...
    ├── values                 # (ΣL, n_groups), (position, sample_set)
    ├── offsets
    ├── contigs
    └── sample_set             # vlen-utf8[n_groups], (sample_set,)
```

## Workflow example: accessibility masks

We provide a working example of converting
[d4](https://github.com/38/d4-format) coverage files into a
genome-wide accessibility mask. Briefly, the steps consist of

1. importing data to a store
2. preprocessing data to compute summaries
3. compute accessibility thresholds
4. export of masks based on thresholds

The examples are based on files stored in the `tests/data` directory
of the pbview repository.

### 1. Data import to datastore

Data import delegates to the pbzarr import engine, which currently
supports `d4` files (direct import from `bam`|`cram` files is under
development).

```bash
pbview import --track-name depth datastore tests/data/s[1-7].per-base.d4
```

Every chunk of the track holds all samples. The number of positions
per chunk is chosen so that a chunk is about 100 MB uncompressed
(`PBVIEW_TARGET_BYTES`), but never below 100,000 positions
(`PBVIEW_MIN_POSITION_CHUNK_SIZE`), which gives chunks above 100 MB
for more than about 250 samples. Set the number of positions with
`--chunk-size`.

### 2. Data preprocessing

In order to calculate accessibility thresholds, we need to process the
`depth` track to generate `sum` and `missingness` summaries. These can
be generated individually with `pbview preprocess sum` and `pbview
preprocess missingness`, or together as follows:

```bash
pbview preprocess all --track-name depth \
    --missing-data-threshold 3 \
    --sampleset-info tests/data/sampleset_info.tsv datastore
```

`--sampleset-info` takes a two-column, whitespace-separated table of
sample and sample set names. Samples that are not listed belong only to
the `ALL` set, and rows for unknown samples are ignored. The grouping
is written to the track, so later commands, including `pbview serve`,
find it in the store.

The `sum` summary adds all coverage data by sample set. Genomic
regions with excessive coverage often correspond to repeats or
collapsed assembly regions, whereas too little coverage may indicate
hard to sequence regions. By default, the output track is called
`<track_name>_sum`.

The `missingness` summary assigns missing data status to individual
bases based on a missing data threshold. The goal is to identify
regions with high missingness. Bases with coverage at or below
`--missing-data-threshold` (default 3) are called missing. The option
can be given several times, for example `--missing-data-threshold 3
--missing-data-threshold 5`, to summarize several thresholds in one
run. The output track is called
`<track_name>_missing_cutoff=<threshold>`.

### 3. Accessibility thresholds

The coverage track and summaries can be interactively explored with
the command `pbview serve datastore`, which prints the address to open
(add `--show` to open a browser window). The app will present tables with
information on accessibility and threshold values that the user can
modify in the interactive plots. The tables also include suggested
thresholds based on recommendations from the literature. When
satisfied, thresholds can be exported from the main page by pressing
the `Export thresholds` button, which will produce a `yaml` file.

(WIP) Alternatively, but not recommended, accessibility thresholds can
be obtained directly with the CLI. This prints the suggested thresholds
and, with `-o`, writes them to a threshold file:

```bash
pbview preprocess thresholds datastore -o thresholds.yml
```

The file holds the values the app starts with: coverage thresholds from
the coverage distribution of each sample set, and `max_missing_samples`
set to half of the samples in the set. The command checks that the
missingness summary for the chosen `--missing-data-threshold` (default
3) exists and stops with an error otherwise. An existing file is only
overwritten with `--force`.

The coverage statistics behind the defaults (mean, median, standard
deviation and a high quantile) are computed over the whole genome. For
genomes with more than 50 million positions they are estimated from
evenly spaced positions instead, which is much faster and moved the
thresholds by less than 0.3% on a 660 Mbp store. Change the limit with
`--max-positions N` (also on `pbview serve`, where the coverage and
missingness pages use it for the current selection and say so), or set
the environment variable `PBVIEW_MAX_POSITIONS`; `--max-positions 0` uses every
position. Masks are always computed from all positions.

### 4. Export of masks

Once a threshold file `thresholds.yml` has been obtained, the final
step is to generate a mask:

```bash
pbview mask generate datastore thresholds.yml -o datastore_mask
```

Note that this will generate a separate output store consisting of
boolean masks. By default each sample set uses the missing-data
threshold recorded in the threshold file. Give
`--missing-data-threshold` (several times for several thresholds) to
generate masks for those thresholds instead; the layers that depend on
the threshold, `accessible` and `excluded_missing`, then get a
`missing_cutoff` dimension.

> [!NOTE]
> Masks can be either *inclusive*, whereby a locus marked with `True`
> means "keep the locus"; *accessibility* refers to such a mask.
> Alternatively, masks are *exclusive*, whereby a locus marked `True`
> is filtered out. By convention, these masks are prefixed with
> `excluded_`.

Finally, a mask can be exported with `pbview mask export`. For a mask
generated for several thresholds, choose one with
`--missing-data-threshold`.

## Running on a remote machine

`pbview serve` listens on `localhost` only, so only a browser on the same
machine can reach it. To run it on a compute node and view it on your
laptop, listen on an address the laptop's connection can reach and
forward the port with ssh, as for a Jupyter notebook:

1. On the compute node: `pbview serve --address 0.0.0.0 --port 8080 STORE`
2. On your laptop: `ssh -L 8080:NODE:8080 user@login-node`, where `NODE`
   is the host name of the compute node.
3. On your laptop, open <http://localhost:8080>.

If you forward a different local port, for example `-L 9000:NODE:8080`,
the browser connects as `localhost:9000` and the server refuses the
websocket connection unless you allow that origin:
`--allow-websocket-origin localhost:9000` (repeatable; `localhost` and the
port served on stay allowed).

> [!WARNING]
> The app has no login. With `--address 0.0.0.0` everyone who can reach the
> node and port can use it and read the data, and the app logs a warning.
> If the cluster lets you ssh to the compute nodes, you can keep the
> default address and jump through the login node instead:
> `ssh -J user@login-node -L 8080:localhost:8080 user@NODE`.
> Earlier versions listened on all interfaces by default.

## Principal component analysis of the samples

How do the samples relate to each other? `pbview pca` (and the PCA page
of the app) runs a principal component analysis of the coverage with the
samples as the features, so each sample gets a coordinate on every
component. Samples that cluster have similar coverage along the genome;
a sample far from its sample set, or two samples that swap places, stand
out.

```bash
pbview pca datastore -o datastore_pca
```

This writes `datastore_pca.scores.tsv` (the loading of each sample on
each component, and its sample sets) and `datastore_pca.variance.tsv`
(eigenvalue and explained variance of each component).

- The data are the mean coverage of every sample in non-overlapping
  windows of `--window-size` positions (default 10,000, or
  `PBVIEW_PCA_WINDOW_SIZE`) inside the contigs. Contigs shorter than
  the window have none, and the last positions of a contig that do not
  fill a window are not used.
- Every sample is divided by its mean coverage over the windows, so the
  first component is not the sequencing depth, and then centred.
- The contigs and samples can be restricted with `--min-contig-length`,
  `--max-contig-length` and `--sample-set` (repeatable).
- The computation is out of core: one pass over the data with Dask,
  reading only the chunks that hold windows, and only the
  samples x samples covariance is kept in memory, so the size of the
  genome does not matter for the memory.

In the app the PCA runs when you press **Run PCA** (it reads the whole
selection, which takes a while for a large genome), for the current
contig and sample selection. The page says when the selection or the
window size has changed since the last run.

## Releases

The version comes from the git tag (`hatch-vcs`): tag `0.1.0-alpha.3`
gives version `0.1.0a3`. The package is pure Python, so one wheel,
`pbview-<version>-py3-none-any.whl`, serves every platform; its
dependencies, including `pbzarr`, are installed from PyPI.

Publishing a GitHub release runs the `Release` workflow, which builds
the wheel and an sdist from the tag, refuses a build whose version is
not that of the tag and attaches both files to the release. It also
uploads them to PyPI once the repository variable `PUBLISH_TO_PYPI` is
`true`. To build the files locally, check out the tag and run
`pixi r build`; they are written to `dist/`. Anything but a tagged
commit gives a dev version.

Run the workflow by hand (Actions, Release, Run workflow) to add files
to a release that already exists. Give its tag and choose where to
publish: `none` (only attach to the release), `testpypi` or `pypi`.

PyPI is told to trust the workflow (trusted publishing), so no token is
stored. Set this up once:

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a
   pending publisher: project `pbview`, owner `percyfal`, repository
   `pbview`, workflow `release.yml`, environment `pypi`. Do the same on
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
   with the environment `testpypi`. Both need an account first.
2. On GitHub (Settings, Environments) create the environments `pypi`
   and `testpypi`. Requiring a reviewer for `pypi` makes every release
   wait for an approval before it is uploaded.
3. Try the first upload on TestPyPI: run the workflow by hand with
   `publish` set to `testpypi`, then install from there with
   `pip install --index-url https://test.pypi.org/simple/
   --extra-index-url https://pypi.org/simple/ pbview==<version>`.

A version can only be uploaded once and the files of a version cannot be
replaced, so check the version before publishing; a release can be
withdrawn (yanked) on PyPI but not reused. Alpha versions are not
installed unless the version is given (`pbview==0.1.0a4`) or `--pre` is
used.

4. When the first upload has worked, set the repository variable
   `PUBLISH_TO_PYPI` to `true` (Settings, Secrets and variables,
   Actions, Variables). From then on every published release is uploaded
   to PyPI. Until it is set, releases only get the wheel and the sdist
   attached, and PyPI is published to by hand.

## Development note

This library is inspired by
[d4explorer](https://github.com/percyfal/d4explorer). `d4explorer`
stores data in individual `d4` files, making data management difficult
and comes with scalability problems. By utilising the virtues of
`Zarr`, pbview is an attempt to address these issues.

Development initially started with a manual adaptation of `d4explorer`
code to use `pbzarr`. Recently, Claude has been used as an assistant
to speed up implementation, but the main direction and ideas are
developed by the author.
