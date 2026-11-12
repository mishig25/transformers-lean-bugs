# TLA+ pass over huggingface_hub concurrency

huggingface_hub main @ `2c61ae8` (2026-11-11). PlusCal specs checked with TLC; every violated property was mapped back
to the code and reproduced on CPU with the real, unpatched library. The only stub is the Hub HTTP endpoint (a small
local fake server, `fakehub.py` / `fakehub_commit.py`, with real-Hub-like latency: HEAD 0.15 s vs 0.11–0.29 s
measured), except finding 2, which downloads a real 268 MB xet file from huggingface.co. GitHub was not searched for
duplicates.

| # | Severity | Bug |
|---|----------|-----|
| 1 | High | Concurrent `local_dir` downloads download each file once per process and delete/truncate files other processes already got back |
| 2 | Medium-high | A failing `upload_folder` cancels every other xet transfer in the process (unrelated downloads in other threads) |
| 3 | Medium | Cache without symlinks (Windows default, `HF_HUB_DISABLE_SYMLINKS=1`): double download, truncated snapshot file returned |
| 4 | Medium-low | `CommitScheduler` drops the final append on coarse-timestamp filesystems |
| 5 | Low (TLC only) | `local_dir` metadata can name the wrong revision after concurrent downloads of different revisions |

## Setup

```bash
pip install -e ./huggingface_hub fastapi uvicorn      # huggingface_hub checked out at 2c61ae8 (hf_xet installed for #2)
curl -LO https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar   # Java 11+
```

- Reproducers: `PY=python ./run_repros.sh` (finding 4 needs a loop-mounted ext2 image with 1 s timestamps — see the
  script; root required for the mount). Recorded outputs: `out_*.txt`; `verify_*.txt` are independent re-runs.
  `APPLY_FIX=1` applies the proposed fix in-process (`fix_localdir.py`, `fix_upload_pipeline_abort.py`).
- Model checking: `TLA2TOOLS=/path/to/tla2tools.jar ./run_tlc.sh`; logs in `results/` (overview `results/SUMMARY.txt`).

## 1. High — concurrent `local_dir` downloads repeat the download and delete files other processes already got
- Trigger: several processes download into one `local_dir` (every DDP rank calling `snapshot_download(local_dir=D)`,
  or `hf download --local-dir` on several nodes sharing a filesystem).
- `_hf_hub_download_to_local_dir` (file_download.py) decides to download from metadata read before the lock (L1340)
  and a file check (L1429). Under the lock (L1503) nothing is re-checked: `paths.file_path.unlink()` then download.
  Metadata is written after the lock is released (L1518). The `cache_dir` path does re-check under its lock (L1284,
  L1992; comment L1244-1248: the lock exists to avoid double downloads). When the file is in the HF cache, the repeat
  re-copies it in place with `shutil.copyfile` (L1487), truncating a file another rank is reading.
- TLA+ `LocalDir.tla`: `ReturnedStaysValid` (once a call returns, the path keeps that content) violated in ~19 steps
  (`ld_same`, `ld_same_lfs`, `ld_copy`); `ld_fixed_*` hold.
- Repro `repro_snapshot_localdir_ddp.py` (4 ranks): `GET requests per file: {'config.json': 4, 'model.safetensors': 4,
  'tokenizer.json': 4}`; ranks returned at 5.1 / 9.3 / 13.5 / 17.7 s and three of them got `FileNotFoundError` on
  `model.safetensors` 0.03–0.05 s after returning (`verify_snapshot_ddp.txt`). `repro_localdir_concurrent.py`: single
  file 2 GETs and the returned path missing for 3.0 s; copy case: returned file size 0 in 3/3 runs.
- Fix: under the lock, re-check whether the file already holds this etag; download/copy to a temp file and
  `os.replace` instead of deleting first; write the metadata before releasing the lock. With `APPLY_FIX=1`: 1 GET per
  file, all ranks done at ~5 s, no failed opens.
- `from_pretrained` uses the shared cache (`cache_dir`), not `local_dir`, so it is not affected.

## 2. Medium-high — a failing `upload_folder` cancels every xet transfer in the process
- `_UploadPipeline.run` (_upload_pipeline.py:451-457) catches any exception (`except BaseException`) and calls
  `abort_xet_session()`, which cancels the single process-wide xet session (utils/_xet.py). Every other caller does so
  only on `KeyboardInterrupt` (file_download.py:601, _commit_api.py:679, hf_api.py:14964/15208). `upload_folder` uses
  this pipeline whenever `hf_xet` is installed (default).
- TLA+ `UploadPipeline.tla` `pipe_bug`: `OtherUnaffected` violated in 5 steps; `pipe_fixed` holds.
- Repro `repro_upload_failure_aborts_downloads.py` (`verify_upload_abort.txt`): upload `FAILED HfHubHTTPError: 403`
  and the concurrent 268 MB download from huggingface.co `FAILED RuntimeError: Operation cancelled: Task cancelled:
  task 12 was cancelled`; control without the upload: OK in 5.2 s; with the fix: upload still 403, download OK in 5.1 s.
  (The first recorded run in `out_upload_failure_aborts_downloads.txt` is marked invalid: it was rate-limited.)
- Fix: global abort only on `KeyboardInterrupt`; otherwise abort only this pipeline's own upload.

## 3. Medium — cache without symlinks: double download, truncated snapshot file
- Windows without Developer Mode (default) or `HF_HUB_DISABLE_SYMLINKS=1`. The first process moves the blob into the
  snapshot folder (L721); a second process waiting on the lock finds no blob (L1284) and downloads again (stray blob).
  For a newer commit with the same file the pointer is created by `shutil.copyfile` into the final path
  (L1260 → L724), which other calls return as soon as it exists (L1100-1113, L1226).
- TLA+ `Cache.tla`: `DownloadOnce` (`cache_nosym`) and `ReturnedOk` (`cache_nosym_ret`, 26 steps) violated.
- Repro `repro_nosymlink_partial.py`, 3/3 runs: 2 GETs; a reader got 65,536 / 589,824 / 983,040 bytes of 268,435,456.
- Fix: copy to a temp file and `os.replace`; under the lock skip the download if the pointer already exists.

## 4. Medium-low — `CommitScheduler` drops the final append on coarse-timestamp filesystems
- A file counts as changed only if `last_uploaded != st_mtime` (_commit_scheduler.py:238, recorded at L244/L276); an
  append in the same timestamp tick right after a scheduled scan is never uploaded, including by the final push in
  `__exit__`.
- TLA+ `CommitScheduler.tla`: `FinalFlush` violated with coarse mtime (`cs_coarse`), holds with fine/multigrain.
- Repro `repro_commit_scheduler_lost_write.py` on ext2 (1 s timestamps): `local lines=2 remote lines=1 -> LOST WRITE`
  (deterministic), 1/10 random runs lost lines; on ext4 0/10. Not on Linux ≥ 6.13 (multigrain timestamps); affects
  FAT/exFAT, HFS+, some network filesystems, older kernels.
- Fix: also compare size (the scheduler assumes append-only files), or track `(st_mtime_ns, st_size)`.

## 5. Low (TLC only) — `local_dir` metadata can describe the wrong revision
Two processes download different revisions into the same `local_dir`; the later metadata write (outside the lock) wins,
so metadata names revision 1 while the file holds revision 2, and a later `hf_hub_download(revision=<commit1>)` returns
revision 2's content via the commit-hash shortcut (L1343). `TerminalMetaSound` violated in `ld_diff`; fixed by fix 1.

## What TLC verified (bounds) and assumptions
- Shared cache with symlinks, with/without shared blob store: 4 processes, 2 commits sharing an etag — no truncated or
  dangling file returned, one download, no deadlock (9,440 states).
- Upload pipeline: 2–3 batches with commit and coordinator failures — no deadlock, termination, no batch committed
  twice, a normal return means every batch was committed.
- `CommitScheduler` with fine/multigrain timestamps: 2 appends, 2 scheduled pushes — no lost write at exit.
- Assumptions: rename/`os.replace` atomic; file locks honoured; each locked region is one step; network calls touch no
  shared state; the xet session is modelled as a counter an abort bumps; HTTP retries not modelled.
