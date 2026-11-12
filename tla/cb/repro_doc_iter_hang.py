"""Reproducer: the first ContinuousBatchingManager example of docs/source/en/continuous_batching.md never terminates.
`for result in manager:` (ContinuousBatchingManager.__iter__, continuous_api.py:1033-1038) loops while the generation
thread is alive, but the thread only stops when the `with` block exits, i.e. after the loop.

TLA+ counterexample: CBOutput.tla / out_doc.cfg, property DocTerminates.
"""
import threading
import time

import torch

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.generation.configuration_utils import ContinuousBatchingConfig

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None,
                  pad_token_id=None)
model = LlamaForCausalLM(cfg).eval()
generation_config = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=8)
cb_cfg = ContinuousBatchingConfig(num_blocks=64, page_size=4, use_cuda_graph=False, use_async_batching=False,
                                  auto_switch_to_flash=False)
seen = []


def docs_example():  # verbatim structure of the docs example (token ids instead of a tokenizer)
    with model.continuous_batching_context_manager(generation_config=generation_config,
                                                   continuous_batching_config=cb_cfg) as manager:
        manager.add_request(input_ids=[1, 2, 3, 4, 5], request_id="long", max_new_tokens=16)
        manager.add_request(input_ids=[6, 7], request_id="short_0", max_new_tokens=4)
        manager.add_request(input_ids=[8, 9, 10], request_id="short_1", max_new_tokens=4)
        for result in manager:
            seen.append(result.request_id)
            print(f"[{result.request_id}] {result.generated_tokens}")
    seen.append("<context exited>")


t = threading.Thread(target=docs_example, daemon=True)
t.start()
t.join(timeout=15)
print("results received:", [s for s in seen if not s.startswith("<")])
print("example still running after 15s (all results were received long before):", t.is_alive())
