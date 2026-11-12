"""Reproducer: stop(hard_stop=True) silently drops requests that were queued but not yet admitted by the loop.

TLA+ counterexample: CBLoop.tla / sync_hardstop.cfg, invariant NoLostOnHardStop.

Docs (docs/source/en/continuous_batching.md, "Shutting down the manager"): "Pass hard_stop=True to abandon pending
work immediately. Queued and active requests are failed with a RuntimeError instead of finishing."

_update_tp_group_state (continuous_api.py:420-437) drains the input queue *before* reading the stop status, and on
HARD_STOP returns early, discarding the drained requests. _fail_all_remaining_requests then finds an empty input queue,
so those requests get no output at all: no result, no error.
"""

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
model.model.register_forward_pre_hook(lambda *_: time.sleep(0.3))  # a forward pass takes 300 ms

gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=50)
cb_cfg = ContinuousBatchingConfig(
    num_blocks=64, page_size=4, max_batch_tokens=64, use_cuda_graph=False, use_async_batching=False,
    auto_switch_to_flash=False,
)
manager = model.init_continuous_batching(generation_config=gen_cfg, continuous_batching_config=cb_cfg)
manager.start()
manager.add_request([1, 2, 3, 4], request_id="A")
time.sleep(0.5)  # A is being generated; the loop is inside a forward pass
manager.add_request([5, 6, 7, 8], request_id="B")  # queued, will be drained at the next iteration
manager.stop(block=True, timeout=10, hard_stop=True)

outputs = []
while not manager.output_router.output_queue.empty():
    outputs.append(manager.output_router.output_queue.get_nowait())
print("outputs after hard stop:", [(o.request_id, o.status.name, o.error) for o in outputs])
ids = {o.request_id for o in outputs}
if "B" not in ids:
    print("BUG: request 'B' was accepted (add_request returned its id) but got no output at all: neither failed nor run")
