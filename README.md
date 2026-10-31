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

### 2. Data preprocessing

In order to calculate accessibility thresholds, we need to process the
`depth` track to generate `sum` and `missingness` summaries. These can
be generated individually with `pbview preprocess sum` and `pbview
preprocess missingness`, or together as follows:

```bash
pbview preprocess all --track-name depth \
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
regions with high missingness. Bases with coverage at or below 3 are
currently called missing; the threshold cannot yet be changed from the
CLI. The output track is called
`<track_name>_missing_cutoff=<threshold>`.

### 3. Accessibility thresholds

The coverage track and summaries can be interactively explored with
the command `pbview serve datastore`. The app will present tables with
information on accessibility and threshold values that the user can
modify in the interactive plots. The tables also include suggested
thresholds based on recommendations from the literature. When
satisfied, thresholds can be exported from the main page by pressing
the `Export thresholds` button, which will produce a `yaml` file.

(WIP) Alternatively, but not recommended, accessibility thresholds can
be obtained directly with the CLI:

```bash
pbview preprocess thresholds datastore
```

### 4. Export of masks

Once a threshold file `thresholds.yml` has been obtained, the final
step is to generate a mask:

```bash
pbview mask generate datastore thresholds.yml -o datastore_mask
```

Note that this will generate a separate output store consisting of
boolean masks.

> [!NOTE]
> Masks can be either *inclusive*, whereby a locus marked with `True`
> means "keep the locus"; *accessibility* refers to such a mask.
> Alternatively, masks are *exclusive*, whereby a locus marked `True`
> is filtered out. By convention, these masks are prefixed with
> `excluded_`.

Finally, a mask can be exported with `pbview mask export`.

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
