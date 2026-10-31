# Issues

## CLI

- [ ] missingness option for `pbview preprocess missingness` and
      `pbview preprocess all` missing; add to `pbview.cli.preprocess`
      (currently the default cutoff 3 cannot be changed).
      `preprocess()` already takes `missing_cutoff`, so this is only
      plumbing
- [ ] add option to export `thresholds.yml` via `pbview preprocess
      thresholds` (needs a default for `max_missing_samples`, which
      only the UI computes today as n_samples / 2, and a profile
      builder that does not require a UI state)
- [ ] `pbview preprocess thresholds --threshold INT` should make use
      of `depth_missing_cutoff=INT`; throw error if this has not been
      preprocessed. Also, `--threshold` is a poor name (better
      `--missing-data-threshold`?; meaning is "Coverage threshold for
      calling a base as present or missing data)
      Today `--threshold` has no effect: `compute_thresholds` accepts
      `missing_cutoff` but never uses it. The check is
      `track.summaries.missingness.get(INT)`; `generate_mask` has an
      error message to reuse. Use one option name across all
      preprocess commands and the README
- [ ] `pbview serve` opens a browser by default (`--show`); consider
      `--no-show` as the default
- [ ] `pbview summarize` and `pbview annotation` have not been
      exercised since the selection refactor
- [ ] stores preprocessed before the summary attrs
      (`perbase:source_track`, `perbase:missing_cutoff`) need
      `pbview preprocess` again; document, or add a clear error

## Other

- [ ] coverage above `PBVIEW_MAX_COVERAGE_PER_SAMPLE` per sample is
      truncated into the last histogram bin; only a log warning says so
- [ ] `Track.from_tree(raw=None)` fails; make `raw` required
