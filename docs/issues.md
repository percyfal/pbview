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
      exercised since the selection refactor
- [ ] stores preprocessed before the summary attrs
      (`perbase:source_track`, `perbase:missing_cutoff`) need
      `pbview preprocess` again; document, or add a clear error

## Other

- [ ] coverage above `PBVIEW_MAX_COVERAGE_PER_SAMPLE` per sample is
      truncated into the last histogram bin; only a log warning says so
- [ ] `Track.from_tree(raw=None)` fails; make `raw` required
