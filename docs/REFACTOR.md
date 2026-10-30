# Refactor plan

Migrating `Coordinates` from legacy boolean masks to pure selection
values (`SpatialSelection`, `ColumnSelection`) resolved against their
universes (`Genome`, `Track`). Each step follows expand → migrate →
contract, with a green suite at every commit.

## Conventions

- `# SHIM`: temporary code; deleted in the contract commit of its
  step. Say what replaces it: `# SHIM: callers move to reset()`.
- `REFACTOR(<step>)`: tag in code, `xfail` reasons and stubs, using
  the step names below. `git grep -n "REFACTOR(samples)"` must be
  empty when a step closes.
- `ShimWarning` (`pbview._compat`) is emitted only at the boundary of
  the legacy layer (public legacy accessors), never in plumbing,
  `__repr__`, `__post_init__` or final-API methods. Shim internals use
  private names.
- Known-broken tests: `xfail(strict=True, raises=...,
  reason="REFACTOR(<step>): ...")`, never `skip`. Legacy-mechanism
  tests (masks, parity, warnings) are tagged `# SHIM` and deleted on
  contract.
- Final-behavior tests use the `coords` / `genome` fixtures, never
  `legacy_coords` directly.

## Commands

- Suite: `pixi r test`
- All shim callers: `pixi r test -- -W error::pbview._compat.ShimWarning`
- One area: `pixi r test -- -W
  "error:Coordinates.sample:pbview._compat.ShimWarning" -q -rf
  --tb=line`
- Types: `basedpyright src tests`

## Steps

### 1. contigs: DONE

Contig mask (`_contig_selected`) replaced by `SpatialSelection` +
`Genome`.

- `Genome.contig_indices` validates, dedups, returns genome order.
- `Genome.resolve` returns merged global `PositionIntervals`
  (`_merge_adjacent`).
- `Genome.contig_regions` returns contig-local `(name, 0, length)`;
  `to_pyranges` lives at the edge, not in `pbview.model`.
- `None` = all contigs, `()` = none. Empty → `None` happens only in
  the shells.
- Contig validation happens at resolve time, not construction.
- Leftovers belonging to later steps: `_select_positions` bridge
  (track-select), `coords.datastore.genome` reads (datastore).

### 2. track-meta: DONE

- `Track.from_tree(name, pb_group, raw_tree)` locates `/{name}/_meta`.
- `meta` keyword-only, separate from legacy `preprocess_data`.
- `track.grouping`: DataArray `/{track}/_meta/column_grouping`, dims
  `(sample, sample_set)`. Dim is singular `sample_set`; assert this in
  tests.
- Prerequisite for samples.

### 3. samples: IN PROGRESS

Sample mask (`_sample_selected`) replaced by `column:
tuple[ColumnSelection, ...]` resolved by the track.

Every track exposes a derived identity grouping `(<column_dim>,
identity)` with the set `ALL` (`IDENTITY_DIM`,
`DEFAULT_GROUPING_SET`), never stored. Stored groupings are
user-supplied only, always contain an all-True `ALL`
(`ensure_all_set`), and cannot use the dim `identity`. A summary's
column dim is its source grouping's group (`identity` when there is no
user grouping), and `perbase:source_grouping_hash` detects staleness.
A track whose column dim is `identity` (a summary of the identity)
carries no derived grouping.

`compute_track_sum` and `compute_track_missingness` now use the
membership and write the `perbase:*` attrs.

- [x] `ColumnSelection` owns its invariants (tuple labels, rejects
      str, `()` allowed)
- [x] `column` is a hashable tuple; `column_for(dim)`; `_with_column`
      replaces by dim
- [x] `Coordinates.__post_init__` rejects bare `ColumnSelection` and
      duplicate dims
- [x] pure `resolve_columns(selections, column_dim, universe,
      grouping)` + tests
- [x] `Track.resolve_columns(selections)`; `Coordinates.resolve_for`
      composes
- [x] decide and pin both-dims policy (proposed: intersect)
- [x] `__post_init__` derives `_sample_selected` from `column` (no
      `_sync_sample_mask`)
- [x] parity test (`# SHIM`); fix `sample_selected` warning text →
      `resolve_for(track, genome)`
- [x] migrate: `_key()` uses column, `matching_sample_set` reads
      column
- [ ] migrate: `_filtered` / `SelectionStateBase`, `Track._select`
      column half (bridge + `_sel_columns`), `compute_track_sum` via
      `(coverage * membership).sum("sample")`
- [x] grouping: `validate_grouping(s)`, `membership_from_sets`,
      `write_column_groupings`
- [x] layout: /{track}/_meta/column_groupings (Dataset, one variable
      "<group_dim>_membership" per group dim)
- [x] preprocess use the writer
- [x] fixtures use the writer; fixtures own groupings, never read back
- [ ] coordinates: `_legacy_grouping` (SHIM) reads
      `groupings["sample_set"]`; reject duplicate column dims
- [ ] contract: delete mask, property, accessors (`samples_all`,
      `n_samples*`, `sample_set_membership_all`, ...), `_replace`,
      `"active": ...` dict builders, shim tests

### 3b. fixtures: TODO

- [x] make use of `pbview.grouping.write_column_groupings` to write
      sample sets
- [x] factor out redundant functions from fixtures (e.g., `_make`
      functions in `conftest.py`

### 4. track-select: TODO

Replace `Track.data(coord)` with `Track.select(resolved:
ResolvedSelection)`. Callers: `track.select(coords.resolve_for(track,
genome))`.

- [ ] Delete `Track.data`, `_select`, `_select_positions` (and the column
  bridge).
- [ ] Keep `_isel_intervals` (final code; works on Dataset and DataArray).
- [ ] Can be done before finishing samples once `resolve_for` exists.
- [ ] convert Track to @dataclass(frozen=True); __post_init__ calls
      validate_grouping

### 5. precomputed: TODO

Remove `PrecomputedTrack`, `preprocess_data`, `Track._pre`.

- [ ] Summaries found via `DataStore.summaries_of(track_name)` using
  `perbase:source_track`.
- [ ] Old stores: rerun `preprocess`; no name-based fallback.
- [ ] Replace `_coverage_hist_cached`'s use of `matching_sample_set` /
  `_pre.sum`.

### 6. datastore: TODO (last)

Remove `Coordinates.datastore`.

- [ ] `reset()` -> `Coordinates()`
- [ ] `ds.base_coord` → `Coordinates()` (many call sites; mechanical).
- [ ] `genome` / `coords` fixtures become literals.
- [ ] Delete custom `__hash__` / `__eq__` / `_key()`; rely on frozen
  dataclass.
- [ ] Delete `pbview/_compat.py` and any `filterwarnings` entry in the
  same commit as the last shim.
- [ ] Move durable decisions to docstrings / docs/design.md; delete
  REFACTOR.md.
- [ ] Check that `Genome` is hashable for the `lru_cache` function
      `SelectionStateBase._filtered`; if `Genome` holds numpy arrays a
      custom `__hash__` function is needed
- [x] DataStore takes sample sets from the track groupings. After that
      sampleset_info goes away on DataStore and on serve, mask,
      annotation and summarize.

## Stubbed / known gaps (app)

- `SelectionStateBase.summary_df`: sample row and sample-set table
  missing → samples
- `selection_overview` stub in the app → samples
- `sample_set_summary`: deleted or stubbed → samples
- `SelectionStateBase` itself is legacy; when replaced, `_filtered`
  moves with the UI state (its body is already final API) → after
  datastore

## Open questions

- Both `sample` and `sample_set` selected: intersect? (pin with test)
- Order: finish samples first, or do track-select first to shrink it?
- Rename `pbview._compat` → `pbview._shims`?
- `tests/conftest.py` still calls `preprocess.preprocess(p,
  sampleset_info=...)` (old signature).
- Move grouping_hash into grouping.py with the preprocess migration.
  Today only the test helpers compute it.
- Preprocess must write summaries named by their source dim.
- A CLI to update or overwrite a user grouping.
- Rename the option to --sampleset-info and keep it on preprocess
  only, plus a dedicated command to add or update a track's grouping.
- Remove the select_set shim (REFACTOR(precomputed)).
- Overlapping sets raise until the `Coordinates` migration

## Contract checklist (every step)

1. `git grep -n "# SHIM"` and `git grep -n "REFACTOR(<step>)"`:
   nothing left for this step
2. `pixi r test` and `pixi r test -- -W
   error::pbview._compat.ShimWarning`
3. `pixi r lint`
4. `git grep -nw` on every removed name: empty
5. Start the app and run the CLI once; exercise the affected widgets
