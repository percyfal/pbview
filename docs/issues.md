# Issues

## ClI

- [ ] missingness option for `pbview preprocess missingness` missing;
      add to `pbview.cli.preprocess` (currently default missingness=3
      cannot be changed)
- [ ] add option to export `thresholds.yml` via `pbview preprocess
      thresholds`
- [ ] `pbview preprocess thresholds --threshold INT` should make use
      of `depth_missing_cutoff=INT`; throw error if this has not been
      preprocessed. Also, `--threshold` is a poor name (better
      `--missing-data-threshold`?; meaning is "Coverage threshold for
      calling a base as present or missing data)
