"""Reproducer (low priority): with two threads each iterating `request_id_iter` over their own streaming request, the
requeue in `get_result(request_id=...)` (continuous_api.py:1026-1028: pop the head, put it back at the TAIL if it
belongs to someone else) reorders a request's streamed chunks: a consumer can see chunk k+1 before chunk k, and may
return on the finished chunk while an older chunk of its request stays in the shared queue forever.

TLA+ counterexample: CBOutput.tla / out_rid3.cfg (invariant InOrder) and out_rid.cfg (QueueDrained, RidGetsFinal).

The only interleaving help is a 20 ms pause before the requeue `put` (the consumer thread being descheduled between
its `get` and its `put`), injected by wrapping the queue's put method for non-generation threads.
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
model.model.register_forward_pre_hook(lambda *_: time.sleep(0.01))
gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=12)
cb_cfg = ContinuousBatchingConfig(num_blocks=64, page_size=4, use_cuda_graph=False, use_async_batching=False,
                                  auto_switch_to_flash=False)
manager = model.init_continuous_batching(generation_config=gen_cfg, continuous_batching_config=cb_cfg)

q = manager.output_router.output_queue
orig_put = q.put


def put(item, *args, **kwargs):
    if threading.current_thread() is not manager._generation_thread:  # a consumer requeueing someone else's output
        time.sleep(0.02)
    return orig_put(item, *args, **kwargs)


q.put = put
manager.start()
seen = {"A": [], "B": []}


def consume(rid):
    for chunk in manager.request_id_iter(rid):  # the documented streaming loop
        seen[rid].append(len(chunk.generated_tokens))


manager.add_request([1, 2, 3, 4], request_id="A", streaming=True)
manager.add_request([5, 6, 7, 8], request_id="B", streaming=True)
threads = [threading.Thread(target=consume, args=(rid,)) for rid in ("A", "B")]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=30)
time.sleep(0.2)
for rid, lens in seen.items():
    print(f"{rid}: chunk lengths seen = {lens}  in order: {lens == sorted(lens)}  got all 12: {sorted(lens) == list(range(1, 13))}")
print("stale outputs left in the shared queue:", [(o.request_id, len(o.generated_tokens)) for o in list(q.queue)])
manager.stop(block=True, timeout=5)
