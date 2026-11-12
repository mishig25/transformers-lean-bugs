# TLA+ pass over datasets concurrency / distribution / resume

datasets main @ `efc8247` (2026-11-10). PlusCal specs checked with TLC; every violated property was mapped back to the
code and reproduced on CPU with the real, unpatched library (local parquet/json/tar files, temp caches; torchdata 0.11.0
for `StatefulDataLoader`). Proposed fixes were validated by monkeypatching them in (`py/fix_patch.py`,
`py/check_fixes.py`): every reproducer and sweep then passes. GitHub was not searched for duplicates.
Line numbers are in `src/datasets/iterable_dataset.py` unless noted.

| # | Severity | Bug |
|---|----------|-----|
| 1 | High | Resuming `StatefulDataLoader` with `num_workers>=2` crashes (`IndexError`/`KeyError`) for `.shuffle()`d streams and some `concatenate_datasets` |
| 2 | High | A loaded state is never cleared (every later epoch restarts at the checkpoint) and `state_dict()` doesn't reflect it (checkpoint right after resume replays data) |
| 3 | High (silent) | Resuming a shuffled arrow-backed stream replays already-trained examples; DDP ranks resume from different positions |
| 4 | Medium | `map(batched=True, drop_last_batch=True[, num_proc])` returns unprocessed rows |
| 5 | Medium-low | Concurrent extraction of the same archive gives readers partial data (`utils/extract.py`) |

## Setup

```bash
pip install -e ./datasets torch torchdata==0.11.0      # datasets checked out at efc8247; CPU torch is enough
curl -LO https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar   # Java 11+
```

- Reproducers: `python py/<repro>.py`; recorded outputs next to them (`py/*.out`). Sweeps: `py/sweep_*.py`
  (`*.patched.out` = with the fixes from `py/fix_patch.py`).
- Model checking: `TLA2TOOLS=/path/to/tla2tools.jar ./run_all.sh`; results in `tlc_results.txt`.

## 1. High — resuming StatefulDataLoader with `num_workers>=2` crashes for `.shuffle()` or `concatenate_datasets`
- `_iter_pytorch` (2909) calls `_prepare_ex_iterable_for_iteration()`, which loads the worker's saved state into the
  full, unsharded tree (3051-3054) before the tree is split per worker and loaded again (2944-2947). The state loader
  (300-313) indexes the saved list with the target's list lengths, so it fails when the saved list is shorter:
  `.shuffle()` (default `max_buffer_input_shards=10`, 4061-4066) builds `min(local shards, 10)` entries (1971-1985);
  `concatenate_datasets` holds one entry per dataset but a worker's tree one per shard it got (1109-1121).
- Trigger: any shuffled stream with 2 ≤ num_shards < 10 × num_workers, or concatenating two 1-shard datasets; with or
  without `split_dataset_by_node`. Unshuffled datasets (the case the existing test covers) are fine.
- TLA+ `LoaderResume.tla` (`LoaderResume_crash.cfg`), NoCrash: shuffle, 2 shards, 2 workers → checkpoint → restore →
  worker loads into the 2-entry tree → crash.
- Repro `py/repro_resume_workers_crash.py` (re-run independently): 4 files unshuffled OK;
  `.shuffle(seed=42, buffer_size=8)`, `.shuffle()+split_dataset_by_node`, and concat of two 1-file datasets →
  `RESUME FAILED: IndexError: list index out of range` (iterable_dataset.py:2909).
- Fix: in `_iter_pytorch`, don't load the starting state in the unsharded prepare; load it only into the worker tree.

## 2. High — a loaded state is never cleared, and `state_dict()` doesn't reflect it
- 2a: `load_state_dict` (2867) only stores the state; 3051-3054 / 2944-2947 re-apply it on every iteration and nothing
  clears it. With `num_workers=0` or `persistent_workers=True` (and no `set_epoch`), every later epoch restarts at the
  checkpoint. Repro `py/repro_sticky_resume_epochs.py` (re-run independently): epochs 0, 1, 2 after resume all
  `[4, 5, 6, 7]` of 0–7; non-persistent workers give the full `[0..7]` from epoch 1, confirming it's unintended.
- 2b: `state_dict()` returns `_state_dict` (2817), which `load_state_dict` does not update; StatefulDataLoader uses it as
  each worker's state right after resuming. Repro `py/repro_resume_then_checkpoint.py` (2 workers): a checkpoint 1 step
  after resuming duplicates `[4, 5, 6, 7]`; at 0 steps it duplicates `[0..7]`.
- TLA+ `LoaderResume.tla`: LaterEpochsComplete (`_sticky`) and ReportedAfterLoad (`_report`) violated; `_fixed` holds
  all four properties (5.35M states).
- Fix: clear `_starting_state_dict` once the iteration that uses it starts; `load_state_dict` also sets
  `self._state_dict = _fast_copy(state_dict)`.

## 3. High (silent) — resuming a shuffled arrow-backed stream replays examples; DDP ranks diverge
- Within a rank: a top-level rebatcher (added at 3030-3033) saves "position before the current shuffled table + rows
  consumed" (574-634); on resume it replays that table and skips rows (550-591), but the shuffle buffer's arrow path
  (2100-2135) cannot be replayed (empty buffer, RNG restarts), so already-yielded rows come back and unseen ones are
  skipped. The docs only allow examples in the shuffle buffer to be lost.
- Across ranks: the `examples` split strategy (`StepExamplesIterable._iter_arrow`, 760-769) consumes whole shuffled
  tables, so ranks that yielded the same k examples resume from different source positions.
- TLA+ `ShuffleResume.tla` NoReplay (N=5, buffer 2: yield 1 → checkpoint → resume yields 1 again); python path and fix
  hold up to N=8. `StepResume.tla` RanksAligned (saved positions 0 vs 8 across ranks); element-wise path holds.
- Repro `py/repro_shuffle_resume_duplicates.py` (parquet, `buffer_size=1000`, re-run independently): 130 examples
  replayed at k=3000 via `ds.state_dict()`; checkpoints every 250 examples replay in 35/36 cases (mean 104, max 137).
  `py/repro_ddp_shuffle_resume.py`: 32 shards / 4 ranks → 540 within-rank replays; 30 shards / 4 ranks → 36 replays
  and 93 examples served to two ranks.
- Fix: above a shuffle buffer, the rebatcher saves the child's live state after the table is produced with 0 rows to
  skip (rest of the table counts as lost buffer); `StepExamplesIterable` uses its element-wise path when a shuffle
  buffer sits below it. Patched: 0 replays, 0 cross-rank duplicates (up to ~half a buffer of extra loss).

## 4. Medium — `map(batched=True, drop_last_batch=True[, num_proc])` returns unprocessed rows
- A shard (or whole dataset) with fewer rows than `batch_size` never starts the writer; `update_data` stays `None`
  (arrow_dataset.py:3758), so the input shard is returned unchanged (4070) and concatenated with mapped shards
  (3646-3657).
- TLA+ `MapCache.tla` (`MapCache_droplast.cfg`) OutputAllMapped: N=7, B=3, num_proc=3 → `<<-1,-2,-3,4,5,6,7>>`.
- Repro `py/repro_map_drop_last.py`: n=10, num_proc=3 → `[0,10,20,30,4,5,6,7,8,9]`; n=9, num_proc=3 → whole dataset
  unmapped; n=3, num_proc=None → `[0,1,2]` instead of empty.
- Fix: a shard that produces no batch contributes 0 rows, never the input shard.

## 5. Medium-low — concurrent extraction of the same archive gives readers partial data
- `utils/extract.py`: the "already extracted?" check (42-54) runs without the lock; under the lock (448-462) the output
  is deleted and extracted in place. Trigger: concurrent `load_dataset` with different builder configs on one tar in a
  shared cache (same-config loads, e.g. every DDP rank, are serialised by the builder lock).
- TLA+ `Extract.tla` NoTornRead (7 steps); re-check under the lock + extract to temp + rename passes.
- Repro `py/repro_extract_race.py` (4 tar members, 4M rows): 3/5 runs the second process silently got 2M or 3M rows,
  and that truncated dataset is then cached.
- Fix: re-check under the lock, extract to `<out>.incomplete`, rename into place.

## What TLC verified (bounds) and assumptions
- `DistShards.tla`, exactly-once across ranks × workers: holds over 2.3M states (auto/shards/examples strategies, nested
  splits, shard-order permutations, per-worker buffer order, 1–6 shards, world sizes ≤ 9, 0–3 workers); the unseeded
  shuffle variant fails, matching the documented warning.
- `MapCache.tla`, ranks mapping concurrently into a shared cache with `num_proc`: ranks always agree (N=7, B=2,
  num_proc=3, 2 ranks).
- Python sweeps on unpatched main: exactly-once 1344 + 9440 configurations, 0 violations; resume 2782 + 1738
  checkpoints (surfaced 1 and 3); double resume 1298 cases (only 2b).
- Assumptions: RNG permutations nondeterministic (over-approximation); shard orders restricted to identity / reverse /
  rotation; each shuffled table produced atomically; state trees reduced to the list lengths the loader indexes;
  torchdata 0.11.0 worker-loop semantics; same `num_proc` on all ranks; deterministic map function.
