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
- [ ] `pbview serve` opens a browser by default (`--show`); consider
      `--no-show` as the default
- [ ] `pbview summarize` and `pbview annotation` have not been
      exercised since the selection refactor; hidden for the time
      being
- [x] INVALID: no user base. stores preprocessed before the summary
      attrs (`perbase:source_track`, `perbase:missing_cutoff`) need
      `pbview preprocess` again; document, or add a clear error

## Other

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
