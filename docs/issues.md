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

- [ ] only the threshold defaults can be estimated from a subset of
      positions (`--max-positions`); `contig_mean_coverage`, the
      per-selection histograms and `missingness_hist` still read the
      whole genome, once per sample set

- [x] coverage above `PBVIEW_MAX_COVERAGE_PER_SAMPLE` per sample is
      clipped into the last bin of `Track.sum_hist` (and of
      `coverage_hist` at `maxbins`), with a log warning; the warning
      claimed this before `sum_hist` actually did it
- [x] `Track.from_tree(raw=None)` fails; make `raw` required. Fix:
      accept `None`, return track with identity grouping only
