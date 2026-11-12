"""Companion to repro_async_cancel_keyerror.py: cancel a request at a random moment while it is *decoding* (async
batching, prefix sharing on). TLA+ predicts a crash iff the in-flight decode step completes a page, i.e. with
probability ~1/page_size per cancellation (page_size=4 here, 256 by default)."""
import random
import time

import torch

import transformers.generation.continuous_batching.input_outputs as io_mod
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.generation.configuration_utils import ContinuousBatchingConfig


class CPUStream(torch.cpu.Stream):
    def synchronize(self): pass
    def record_event(self, event=None): return event
    def wait_event(self, event): pass


io_mod.ContinuousBatchingAsyncIOs._supported_device_types = ("cuda", "xpu", "cpu")
io_mod.create_device_stream = lambda device: CPUStream()

torch.manual_seed(0)
random.seed(0)
cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None,
                  pad_token_id=None)
model = LlamaForCausalLM(cfg).eval()
model.model.register_forward_pre_hook(lambda *_: time.sleep(0.01))
gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=200)
crashes, trials = 0, 40
for trial in range(trials):
    cb_cfg = ContinuousBatchingConfig(num_blocks=256, page_size=4, max_batch_tokens=64, use_cuda_graph=False,
                                      use_async_batching=True, auto_switch_to_flash=False)
    manager = model.init_continuous_batching(generation_config=gen_cfg, continuous_batching_config=cb_cfg)
    manager.start()
    manager.add_request([1, 2, 3], request_id="innocent", max_new_tokens=200)
    manager.add_request([5, 6, 7], request_id="victim", max_new_tokens=200)
    time.sleep(0.15 + random.random() * 0.3)  # well into the decode phase
    manager.cancel_request("victim")
    time.sleep(0.15)
    crashed = manager.background_thread_status.fatal_error is not None
    crashes += crashed
    manager.stop(block=True, timeout=5, hard_stop=True)
print(f"decode-phase cancellations that crashed the manager: {crashes}/{trials} (page_size=4 -> expected ~{trials // 4})")
