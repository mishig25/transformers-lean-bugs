"""Sanity check for the CPU async-IO emulation used by repro_async_cancel_keyerror.py: without cancellation, async
batching on CPU (stubbed streams) must give exactly the same tokens as sync batching and as model.generate."""
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
cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None, pad_token_id=None)
model = LlamaForCausalLM(cfg).eval()
gc = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=12)
prompts = [list(range(1, 6)), list(range(20, 32)), [7, 8, 9], list(range(40, 49))]
outs = {}
for use_async in (False, True):
    cbc = ContinuousBatchingConfig(num_blocks=64, page_size=4, max_batch_tokens=16, use_cuda_graph=False,
                                   use_async_batching=use_async, auto_switch_to_flash=False)
    res = model.generate_batch(prompts, generation_config=gc, continuous_batching_config=cbc, warmup=False)
    outs[use_async] = [res[f"req_{i}"].generated_tokens for i in range(len(prompts))]
ref = [model.generate(torch.tensor([p]), generation_config=gc)[0, len(p):].tolist() for p in prompts]
print("sync == generate :", outs[False] == ref)
print("async == generate:", outs[True] == ref)
