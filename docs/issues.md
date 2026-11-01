# Issues

## CLI

- [x] missingness option for `pbview preprocess missingness` and
      `pbview preprocess all` missing; add to `pbview.cli.preprocess`
      (currently the default cutoff 3 cannot be changed).
      `preprocess()` already takes `missing_cutoff`, so this is only
      plumbing
- [x] add option to export `thresholds.yml` via `pbview preprocess
      thresholds`
      Done: `-o/--output` (and `--force`); `profile_from_defaults`
      gives the thresholds the app starts with, and
      `default_max_missing_samples` (half of the samples) is shared
      with the app
- [x] `pbview preprocess thresholds --threshold INT` should make use
      of `depth_missing_cutoff=INT`; throw error if this has not been
      preprocessed. Also, `--threshold` is a poor name (better
      `--missing-data-threshold`?; meaning is "Coverage threshold for
      calling a base as present or missing data)
      Today `--threshold` has no effect: `compute_thresholds` accepts
      `missing_cutoff` but never uses it. The check is
      `track.summaries.missingness.get(INT)`; `generate_mask` has an
      error message to reuse. Use one option name across all
      preprocess commands and the README
      Done: `--missing-data-threshold` (`--threshold` kept as an
      alias on `thresholds`); the cutoff is validated but not used
      further yet
- [x] `pbview serve` no longer opens a browser by default; `--show`
      does
- [ ] `pbview summarize` and `pbview annotation` have not been
      exercised since the selection refactor; hidden for the time
      being
- [x] INVALID: no user base. stores preprocessed before the summary
      attrs (`perbase:source_track`, `perbase:missing_cutoff`) need
      `pbview preprocess` again; document, or add a clear error

## Other

- [ ] PCA: contigs shorter than the window size, and the positions of a
      contig that do not fill a window, are not used; a sample in
      several sample sets is coloured by its first set (not `ALL`); the
      page computes on a button press, not live

- [x] the contig plot was empty after a session ended: the views were
      built once and shared by all sessions; `serve` now builds them per
      session (the data store is still shared)
- [x] the genome gauges show the size of the active contigs (and their
      fraction of the genome), not the whole genome
- [x] the contig plot shows every contig whatever the selection; the
      `selected` column is computed but not used to dim the inactive
      contigs

- [x] the coverage and missingness pages use `--max-positions` too
      and read the data once for all sample sets; contig means are
      exact and also one pass
- [ ] a selection of samples that is not exactly one sample set sums
      the raw values. The sets of a selection are read in one pass,
      together with their missingness for every cutoff (so the two
      pages share it), and sampled before the sum. On a 660 Mbp store
      with 32 samples a selection change still takes 75-90 s for both
      pages (was 250-330 s), bound by reading the raw chunks. A
      sample-wise chunk layout for the raw track would help
- [x] estimated pages say so: a note, a `sampling` table column and
      "(estimated from every kth position)" in plot titles

- [x] coverage above `PBVIEW_MAX_COVERAGE_PER_SAMPLE` per sample is
      clipped into the last bin of `Track.sum_hist` (and of
      `coverage_hist` at `maxbins`), with a log warning; the warning
      claimed this before `sum_hist` actually did it
- [x] `Track.from_tree(raw=None)` fails; make `raw` required. Fix:
      accept `None`, return track with identity grouping only

## Features

- [ ] Add sample set grouping table as a `pn.Tabulator` widget,
      columns `sample_set`, `samples`, where the `samples` column is a
      comma-separated list of samples in the sample set.
- [ ] Add sample information table as a `pn.Tabulator` widget. If the
      number of samples exceeds a given number (say 20), the table is
      collapsable or spread across separate columns. Columns:
      `sample` + all sample set definitions, possibly column showing
      sample selection status
