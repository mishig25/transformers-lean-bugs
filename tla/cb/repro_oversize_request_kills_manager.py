"""Reproducer: a single request whose prompt + max_new_tokens exceeds the KV-cache capacity kills the whole CB
manager (RuntimeError "No requests can be scheduled and no requests can be offloaded" -> _handle_critical_error),
failing every other in-flight / queued request and refusing all later ones.

TLA+ counterexample: CBLoop.tla / sync_oversize.cfg, invariant NoCrash ("NoRequestCanBeOffloaded").
"""
import time

import torch

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.generation.configuration_utils import ContinuousBatchingConfig

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None,
                  pad_token_id=None)
model = LlamaForCausalLM(cfg).eval()
gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=20)
# cache: 8 blocks x 4 tokens = 32 tokens
cb_cfg = ContinuousBatchingConfig(num_blocks=8, page_size=4, max_batch_tokens=64, use_cuda_graph=False,
                                  use_async_batching=False, auto_switch_to_flash=False)
manager = model.init_continuous_batching(generation_config=gen_cfg, continuous_batching_config=cb_cfg)
manager.start()
model.model.register_forward_pre_hook(lambda *_: time.sleep(0.01))
manager.add_request([5, 6, 7, 8], request_id="too_long", max_new_tokens=40)   # 43 tokens: can never fit
# wait until too_long holds most of the cache, then submit an ordinary request that fits on its own (17 tokens)
while True:
    bp = manager.batch_processor
    st = bp.scheduler.active_requests.get("too_long") if bp is not None else None
    if st is not None and st.generated_len() >= 18:
        break
    time.sleep(0.001)
manager.add_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="innocent", max_new_tokens=10)
results = {}
t_end = time.time() + 20
while time.time() < t_end and len(results) < 2:
    r = manager.get_result(timeout=0.5)
    if r is not None and r.is_finished():
        results[r.request_id] = r
print("NOTE: 'innocent' fits in the cache on its own; it only had to wait for 'too_long'")
for rid, r in sorted(results.items()):
    print(f"{rid}: {r.status.name} tokens={len(r.generated_tokens)} error={r.error!r}")
print("fatal_error:", repr(manager.background_thread_status.fatal_error))
print("later request accepted:", manager.add_request([9, 10], max_new_tokens=2))
manager.stop(block=True, timeout=5)
