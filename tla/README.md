# TLA+ pass over transformers concurrency

transformers main @ `9b6e5e2` (2026-11-12). TLA+ is used only for the concurrent parts (threads, queues, timers, async
GPU work) — continuous batching and `transformers serve`; the index arithmetic is covered by the Lean files in the rest
of this repo. Specs are PlusCal, checked with TLC. Every violated property was mapped back to the code and reproduced in
Python on CPU against main with the real classes (ContinuousBatchingManager, ModelManager, GenerationState,
ChatCompletionHandler, tiny models).

## Setup

```bash
curl -LO https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar   # needs Java 11+
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ./transformers fastapi uvicorn openai httpx rich   # transformers checked out at 9b6e5e2
```

- Check a spec: `java -cp tla2tools.jar pcal.trans Spec.tla && java -cp tla2tools.jar tlc2.TLC -deadlock -config <cfg> <MC>.tla`.
  `cb/run_all.sh` (with `TLA2TOOLS=/path/to/tla2tools.jar`) re-runs every continuous-batching config; captured TLC
  outputs are in `cb/results/*.txt`.
- Run a reproducer from its folder, e.g. `cd cb && python repro_async_cancel_keyerror.py` or
  `cd serve && python repro_idle_unload_cb.py`. Captured outputs on main: `cb/results/*.out`, `serve/*.out`.
- Async continuous batching needs CUDA; the async reproducers run the real `ContinuousBatchingAsyncIOs` on CPU with a
  no-op stream object (no CB logic patched; `cb/check_async_cpu_fidelity.py` shows identical tokens to sync mode and
  `model.generate`). GPU-specific impacts (OOM) are stated as inferred where they were not run.

## Summary

| # | Severity | Area | Bug |
|---|----------|------|-----|
| 5 | High | continuous batching | async mode: cancelling a request mid-prefill crashes the generation loop (KeyError), failing every request |
| 6 | High | continuous batching | cancelling the last active request after an offload wedges the manager forever |
| 1 | High | serve | `--continuous-batching` + idle unload: model never freed, reload loads a second copy (OOM expected on GPU) |
| 2 | Medium | serve | switching models under CB kills the other model's in-flight streams, blocks the event loop |
| 7 | Medium | continuous batching | `stop(hard_stop=True)` silently drops queued requests |
| 8 | Medium | continuous batching | one request larger than the KV cache kills the manager and every other request |
| 9 | Medium | continuous batching | first ContinuousBatchingManager example in the docs never terminates |
| 3 | Medium-low | serve | streaming CB request hangs forever if it finishes before its handler is registered (latent) |
| 4 | Low-medium | serve | request arriving during the idle unload gets `model=None` (HTTP 500) |
| – | Low | continuous batching | `request_id_iter` with ≥2 consumers reorders / strands chunks |

## transformers serve (`src/transformers/cli/serving/`) — spec [`serve/Serve.tla`](serve/Serve.tla)

Actors: idle-unload Timer thread (TimedModel), request handlers on the event loop, continuous-batching (CB) background
thread + OutputRouter, event-loop callbacks. Invariants: NoFail (handler never gets model=None), OneCopy (never two
copies of a model in memory), NoHang (every SSE stream ends). With `Fixed = FALSE` (current code) TLC violates all
three in ≤ 10 steps; with `Fixed = TRUE` (fixes below) all three hold (3 requests, ≤ 4 copies, full state space).

### 1. High — `--continuous-batching` + idle unload: memory never freed, reload loads a second copy
- `TimedModel.delete_model` (model_manager.py:77-88) drops the cache's reference after `--model-timeout` (default
  300 s), but the CB manager created by `CBGenerateManager.init_cb` (utils.py:851-866) still holds the model (and its
  KV cache). The next request finds the key absent, loads a new copy (model_manager.py:329-345); `get_manager` returns
  the same CB manager for that model id and `init_cb` is a no-op, so generation keeps using the old copy.
- TLA+: OneCopy / CBUsesCached violated (trace: load → timer fires → pop → reload).
- Repro `serve/repro_idle_unload_cb.py` (timeout 5 s instead of 300 s): after the idle period "copy #1 freed? False",
  request 2 loads copy #2, "CB still generates with copy #1: True", 2 live model instances.
- Impact: the documented `transformers serve --continuous-batching` (models loaded per request; not when a model is
  passed positionally, which disables the timeout). On GPU the CB cache takes 80–90% of the memory left after loading
  (`max_memory_percent`), so the reload has ~10–20% headroom: for any model larger than that, the reload is expected to
  OOM and every later request for that model fails until restart (GPU behaviour inferred, not run here).
- Fix: on unload, also stop/drop the CB manager holding that model (or never unload a model with a live CB manager).

### 2. Medium — switching models under `--continuous-batching` kills the other model's in-flight streams
- `GenerationState.get_manager` (utils.py:1047-1065) keeps one CB manager; a request for another model calls
  `stop(block=True, timeout=2)` on the event loop. FLUSH_AND_STOP does not finish within 2 s, `stop()` then sets
  `batch_processor = None` (continuous_api.py:855-856) while the generation thread still runs.
- Repro `serve/repro_cb_model_switch.py`: event loop blocked 3.2 s; "Generation thread did not exit after join timeout
  (2)"; the generation loop crashes ("Tried to perform a generation step before the batch processor was initialized")
  and model A's stream ends with that error. With the default cache sizing the two managers' caches overlap and the
  process was OOM-killed (CPU sizes the cache from free system RAM).
- Fix: one CB manager per model id (as for GenerateManager), or drain/fail the old manager's requests before
  replacing it, off the event loop.

### 3. Medium-low (latent) — streaming CB request can hang forever if it finishes before its handler is registered
- `CBGenerateManager.generate_streaming` (utils.py:902-936) calls `cb.add_request()` before
  `cb.register_result_handler()`. `OutputRouter.deliver` (continuous_api.py:100-108) looks the handler up in the CB
  thread; without one the output goes to the shared `output_queue`, which serve never reads. If that output is the
  finishing one (`max_tokens=1`, immediate EOS), `streamer.end()` never runs and `sse_gen` awaits `queue.get()` forever.
  The non-streaming path registers first, with a comment about exactly this race.
- TLA+: NoHang violated (Add → CB thread delivers → Reg → Wait).
- Repro `serve/repro_cb_stream_register_race.py`: natural: 0/200 concurrent `max_tokens=1` streams hung on CPU;
  with the window widened by 50 ms inside CBStreamer construction: 2/3 streams never end, 2 outputs stranded.
- Fix: register the handler before `add_request` (as the non-streaming path does).

### 4. Low-medium — request arriving while the idle timer unloads the model gets `model=None`
- `delete_model` runs on the Timer thread without the per-model lock, sets `self.model = None`, then runs
  `gc.collect()` + `empty_cache()`, and only then pops the entry. A request in that window takes the cached branch of
  `load_model_and_processor` and returns `None`.
- TLA+: NoFail violated. Repro `serve/repro_timer_vs_request.py` (gc.collect stretched to 1 s): key still cached with
  `.model = None`; request 2 raises `ValueError: Unknown modality for: NoneType` (HTTP 500).
- Fix: take the per-model lock in `delete_model` and pop before clearing (or treat `model is None` as not loaded).

## transformers continuous batching (`src/transformers/generation/continuous_batching/`) — specs `cb/CBLoop.tla`, `cb/CBOutput.tla`, `cb/CBLedger.tla`

Modelled the generation loop (sync and async "prepare N+1, update N"), scheduler, soft-reset offloading, cancellation,
stop/hard stop, prefix sharing and the paged block ledger, plus the output router / iterator APIs. Every finding below
was reproduced on main (CPU, real ContinuousBatchingManager; async mode via a no-op CPU stream object, no CB logic
patched). `cb/run_all.sh` re-runs every TLC config; outputs in `cb/results/`.

### 5. High — in async mode, cancelling a request can crash the whole generation loop
- `Scheduler.clear_cancelled_requests` (scheduler.py:95-113) frees the request's block table but leaves its status.
  The batch already in flight is then applied by `update_batch` (continuous_api.py:566/574) →
  `PagedAttentionCache.mark_complete_blocks` (cache.py:349-357) skips only FINISHED requests →
  `CacheAllocator.mark_complete_blocks` (cache_allocator.py:294) `self.block_table[request_id]` → KeyError.
  The same function's comment covers the offload case ("status can be FINISHED in async mode ... the block table no
  longer exists"), not cancellation.
- TLA+: `async_share_cancel.cfg`, NoCrash violated (r1's page-completing prefill in flight → cancel → next prepare
  frees r1's table → update raises).
- Repro `cb/repro_async_cancel_keyerror.py`: `KeyError: 'victim'`; the unrelated request ends `FAILED`;
  `manager still running: False`; new requests refused. Random cancels during decode crash 14/40 and 11/40 runs
  (`repro_async_cancel_decode_rate.py`, ≈1/page_size per cancel).
- Trigger: async batching is the default when CUDA graphs are on and no attention mask is needed (flash attention,
  initialization.py:280-290); prefix sharing is on by default for full-attention models; `transformers serve` calls
  `cancel_request` on every client disconnect. A disconnect during the prefill of a ≥1-page prompt kills the CB worker
  (serve then answers 503 to everything).
- Fix: mark cancelled states FINISHED in `clear_cancelled_requests` (as soft reset does), or skip requests without a
  block table in `mark_complete_blocks`.

### 6. High — cancelling the last active request after an offload wedges the manager forever
- `offload_requests` sets `scheduler.block_new_requests = True` (offloading_manager.py:236); only a request *finishing*
  resets it (continuous_api.py:569), cancellation never does. With no active request left, the FIFO / prefill-first
  schedulers skip the waiting list (scheduler.py:331, 382) and the loop idles forever with pending requests.
- TLA+: `sync_cancel_atomic.cfg`, liveness `EveryRequestAnswered` / `StopTerminates` violated.
- Repro `cb/repro_cancel_after_offload_hang.py`: `finished after 10s: [] (expected B and C)`,
  `active=[] waiting=['B', 'C'] block_new_requests=True`, flush `stop()` returns only by timeout with the thread alive.
- Trigger: cache pressure (soft reset is the default offload) followed by cancellation/disconnect of the remaining
  active request(s): the server stops serving permanently.
- Fix: reset `block_new_requests` when a cancelled request was active / when no request is active.

### 7. Medium — `stop(hard_stop=True)` silently drops queued requests
- `_update_tp_group_state` drains the input queue (continuous_api.py:422) then returns early on hard stop (:436-437);
  `_fail_all_remaining_requests` finds nothing. Docs say pending requests are failed with a RuntimeError.
- TLA+ `sync_hardstop.cfg` (NoLostOnHardStop). Repro `cb/repro_hard_stop_drops_queued.py`: 'B' accepted, never
  answered (3/3 runs).

### 8. Medium — one request larger than the KV cache kills the manager and every other request
- continuous_api.py:509 raises "No requests can be scheduled and no requests can be offloaded", which fails all
  requests and refuses new ones (docs: offloading happens "rather than crash").
- TLA+ `sync_oversize.cfg` (NoCrash). Repro `cb/repro_oversize_request_kills_manager.py`: an innocent request that fits
  on its own ends FAILED; later requests refused. Fix: fail/reject only the request that can never fit.

### 9. Medium — the first ContinuousBatchingManager example in `continuous_batching.md` never terminates
- `for result in manager:` iterates while the generation thread is alive (continuous_api.py:1033-1038), which only ends
  when the `with` block exits. TLA+ `CBOutput.tla` / `out_doc.cfg` (DocTerminates). Repro `cb/repro_doc_iter_hang.py`:
  all 3 results arrive, still running after 15 s.

### Low
- Serve streaming register race (= finding 3 above), found independently by the CB model (`out_serve.cfg`
  `ServeStreamEnds` fails; `out_serve_fixed.cfg` with the swapped order passes).
- `request_id_iter` with ≥2 concurrent consumers reorders/strands chunks: `get_result(request_id=...)` re-queues other
  requests' outputs at the back (continuous_api.py:1026-1027). TLA+ `out_rid3.cfg` / `out_rid.cfg`; repro
  `cb/repro_request_id_iter_reorder.py` → `[2, 3, 5, 7, 8, 9, 6, 4, 10, 1, 11, 12]`.
- TLC-only (no repro): add_request/stop check-then-put race; streaming output after cancel in async mode (root cause of
  #5); a pause withdrawn in a tiny window could leave the loop paused.

### Verified (TLC)
- Main loop (3 requests, page 2, 4 blocks, prefix sharing, random EOS, streaming, flush stop): all safety invariants
  and both liveness properties hold on main without cancellation/hard stop (3,863 sync / 12,041 async states); with the
  fixes for #5, #6, #7 enabled, cancellation and hard stop included: sync 102,078 / async 425,782 states, all hold.
  `cb/run_with_fixes.py` applies those fixes at runtime: hang repro completes, async crash rate 0/40.
- Block ledger (`CBLedger.tla`, 4 blocks, 3 requests incl. a fork): refcounts, hashes, cached blocks, prefix match,
  fork, dedup, offload, eviction keep 6 invariants over 37,249 states; a seeded bug is caught.
