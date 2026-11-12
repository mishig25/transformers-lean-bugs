"""Reproducer: in async batching mode with prefix sharing (the defaults on CUDA + flash attention for full-attention
models), cancelling a request while its in-flight batch completes a KV page crashes the whole generation loop
(KeyError in CacheAllocator.mark_complete_blocks), failing every other request and killing the manager.

TLA+ counterexample: CBLoop.tla / async_share_cancel.cfg, invariant NoCrash ("KeyErrorInMarkCompleteBlocks").

Async batching normally needs CUDA/XPU streams. CPU kernels run synchronously, so the async bookkeeping (prepare
batch N+1 before update of batch N, double IO buffers, carry-over of tokens) runs unchanged on CPU once we provide a
CPU stream object with a no-op synchronize() and allow "cpu" as an async device. No CB logic is patched.
"""

import threading
import time

import torch

import transformers.generation.continuous_batching.input_outputs as io_mod
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.generation.configuration_utils import ContinuousBatchingConfig


# ---- run ContinuousBatchingAsyncIOs on CPU (infrastructure only) ----
class CPUStream(torch.cpu.Stream):
    # CPU work is synchronous: every stream/event operation is trivially satisfied
    def synchronize(self):
        pass

    def record_event(self, event=None):
        return event

    def wait_event(self, event):
        pass


io_mod.ContinuousBatchingAsyncIOs._supported_device_types = ("cuda", "xpu", "cpu")
io_mod.create_device_stream = lambda device: CPUStream()

torch.manual_seed(0)
cfg = LlamaConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None, pad_token_id=None,
)
model = LlamaForCausalLM(cfg).eval()
# each forward takes 200 ms (stands in for a real model's prefill time); no logic is changed
model.model.register_forward_pre_hook(lambda *_: time.sleep(0.2))

gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=30)
cb_cfg = ContinuousBatchingConfig(
    num_blocks=64, page_size=4, max_batch_tokens=64, use_cuda_graph=False, use_async_batching=True,
    auto_switch_to_flash=False,  # allow_block_sharing=True is the default -> prefix sharing on (full attention)
)
manager = model.init_continuous_batching(generation_config=gen_cfg, continuous_batching_config=cb_cfg)
manager.start()
print("async IO:", type(manager.batch_processor.inputs_and_outputs).__name__ if manager.batch_processor else None)

# An innocent long-running request
manager.add_request(list(range(1, 6)), request_id="innocent", max_new_tokens=30)
time.sleep(0.5)

# A request with a 12-token prompt: its prefill batch completes 3 pages of 4 tokens
manager.add_request(list(range(20, 32)), request_id="victim", max_new_tokens=30)
# The client goes away while the prompt is being prefilled (e.g. HTTP disconnect in `transformers serve`)
time.sleep(0.3)
manager.cancel_request("victim")
print("cancelled 'victim'")

results = {}
t_end = time.time() + 20
while time.time() < t_end and "innocent" not in results:
    r = manager.get_result(timeout=0.5)
    if r is not None and r.is_finished():
        results[r.request_id] = r
    if not manager.is_running():
        break
time.sleep(0.5)
inn = results.get("innocent")
print("innocent:", None if inn is None else (inn.status.name, len(inn.generated_tokens), inn.error))
print("fatal_error:", repr(manager.background_thread_status.fatal_error))
print("manager still running:", manager.is_running())
print("new request accepted after that:", manager.add_request([1, 2, 3], max_new_tokens=2))
manager.stop(block=True, timeout=5)
