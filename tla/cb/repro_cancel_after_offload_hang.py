"""Reproducer: cancelling the last active request after an offload (soft reset) wedges the CB manager forever.

TLA+ counterexample: CBLoop.tla / sync_cancel_atomic.cfg, property EveryRequestAnswered + StopTerminates.

Scenario (sync mode, CPU, default cpu_offload_space=0.0 => offloading == soft reset):
  1. two requests A, B whose combined KV does not fit the cache -> B is starved and soft-reset by
     OffloadingManager.offload_requests, which sets scheduler.block_new_requests = True
  2. the user cancels A (the only remaining active request) with the documented `cancel_request`
  3. block_new_requests is only ever reset when a request *finishes* in update_batch, so it stays True:
     FIFOScheduler.schedule_batch never looks at the waiting list again, returns [] and the loop idles forever.
     B never completes, any new request never completes, and stop() (flush) never returns.
"""

import threading
import time

import torch

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.generation.configuration_utils import ContinuousBatchingConfig


torch.manual_seed(0)
cfg = LlamaConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None, pad_token_id=None,
)
model = LlamaForCausalLM(cfg).eval()

# Slow the forward down a bit (5 ms) so the user thread has time to observe the offload; no logic is changed.
model.model.register_forward_pre_hook(lambda *_: time.sleep(0.005))

gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=40)
# 12 blocks of 4 tokens = 48 tokens of KV. Each request needs 4 + 40 - 1 = 43 tokens = 11 blocks: each fits alone,
# but not both together, so one gets offloaded (soft reset) mid-generation.
cb_cfg = ContinuousBatchingConfig(
    num_blocks=12, page_size=4, max_batch_tokens=64, use_cuda_graph=False, use_async_batching=False,
    auto_switch_to_flash=False,
)

manager = model.init_continuous_batching(generation_config=gen_cfg, continuous_batching_config=cb_cfg)
manager.start()
a = manager.add_request([1, 2, 3, 4], request_id="A", max_new_tokens=40)
b = manager.add_request([5, 6, 7, 8], request_id="B", max_new_tokens=40)

# Wait until the manager offloads a request (observable through the public batch_processor attribute)
deadline = time.time() + 30
while time.time() < deadline:
    bp = manager.batch_processor
    if bp is not None and bp.scheduler.block_new_requests:
        break
    time.sleep(0.0005)
sched = manager.batch_processor.scheduler
print(f"offload happened: active={list(sched.active_requests)} waiting={list(sched.waiting_requests)} "
      f"block_new_requests={sched.block_new_requests}")
remaining_active = list(sched.active_requests)
for rid in remaining_active:
    manager.cancel_request(rid)  # documented API; e.g. `transformers serve` calls it when a client disconnects
print(f"cancelled {remaining_active}")

# A brand new request, submitted after the cancellation, should also complete
c = manager.add_request([9, 10, 11, 12], request_id="C", max_new_tokens=5)

results = {}
t_end = time.time() + 10
while time.time() < t_end and len(results) < 2:
    r = manager.get_result(timeout=0.5)
    if r is not None and r.is_finished():
        results[r.request_id] = r
print(f"finished after 10s: {sorted(results)}  (expected B and C)")
print(f"scheduler: active={list(sched.active_requests)} waiting={list(sched.waiting_requests)} "
      f"block_new_requests={sched.block_new_requests}")

# stop() with the default flush semantics: should wait for B and C then return. Use a timeout to not hang forever.
t0 = time.time()
manager.stop(block=True, timeout=10)
print(f"stop(block=True, timeout=10) returned after {time.time() - t0:.1f}s; generation thread alive: "
      f"{manager.is_running()}")
if manager.is_running():
    print("BUG: the generation loop is wedged: pending requests are never scheduled and flush-stop never completes")
    manager.stop(block=True, timeout=10, hard_stop=True)  # cleanup
    print(f"after hard stop: alive={manager.is_running()}")
