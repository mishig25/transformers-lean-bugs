"""Reproducer: `transformers serve` streaming with continuous batching can lose the final output of a request, so the
SSE stream never ends.

CBGenerateManager.generate_streaming (cli/serving/utils.py:884-937) calls cb.add_request(...) and only afterwards
cb.register_result_handler(...). Any output produced in between goes to the shared output_queue, which serve never
reads. If the request finishes in that window (short max_tokens / early EOS, event-loop thread preempted), the handler
never sees the finished output: the text queue never gets its end sentinel and the client hangs. The non-streaming path
explicitly registers first ("Register future BEFORE add_request to avoid race with fast completion").

TLA+ counterexample: CBOutput.tla / out_serve.cfg, property ServeStreamEnds (fixed order: out_serve_fixed.cfg passes).

Part 1 is deterministic: a 0.2 s pause in CBStreamer.__init__ stands for the event-loop thread being descheduled.
Part 2 has no injected delay: it only runs a few CPU-bound Python threads (GIL contention) and counts hung streams.
"""

import asyncio
import sys
import threading
import time

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import transformers.cli.serving.utils as serve_utils
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.generation.configuration_utils import ContinuousBatchingConfig


torch.manual_seed(0)
cfg = LlamaConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, max_position_embeddings=512, eos_token_id=None, bos_token_id=None, pad_token_id=None,
)
model = LlamaForCausalLM(cfg).eval()
tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")))
tok._tokenizer.pre_tokenizer = Whitespace()

gen_cfg = GenerationConfig(do_sample=False, eos_token_id=-1, max_new_tokens=1)
cb_cfg = ContinuousBatchingConfig(num_blocks=64, page_size=4, use_cuda_graph=False, use_async_batching=False,
                                  auto_switch_to_flash=False)
mgr = serve_utils.CBGenerateManager(cb_config=cb_cfg)
mgr.init_cb(model, gen_cfg)


async def one_stream(rid: str, timeout: float) -> bool:
    """Returns True if the SSE text queue receives its end-of-stream sentinel within `timeout`."""
    q, streamer = mgr.generate_streaming(model, tok, {"input_ids": [1, 2, 3, 4]}, gen_cfg, request_id=rid)
    try:
        while True:
            item = await asyncio.wait_for(q.get(), timeout)
            if item is None:
                return True
    except asyncio.TimeoutError:
        return False


# ---- part 1: deterministic interleaving ----
orig_init = serve_utils.CBStreamer.__init__


def slow_init(self, *args, **kwargs):
    time.sleep(0.2)  # event-loop thread descheduled between add_request and register_result_handler
    orig_init(self, *args, **kwargs)


serve_utils.CBStreamer.__init__ = slow_init
ended = asyncio.run(one_stream("det_0", timeout=5))
serve_utils.CBStreamer.__init__ = orig_init
orphans = list(mgr._cb.output_router.output_queue.queue)
print(f"[part 1] stream ended: {ended}; orphaned outputs in the never-read output_queue: "
      f"{[(o.request_id, o.status.name) for o in orphans]}")

# ---- part 2: no injected delay, only GIL contention from a few busy Python threads ----
stop = threading.Event()


def busy():
    x = 0
    while not stop.is_set():
        x += 1


busy_threads = [threading.Thread(target=busy, daemon=True) for _ in range(3)]
for t in busy_threads:
    t.start()
sys.setswitchinterval(0.005)


async def many(n: int) -> list[str]:
    hung = []
    for i in range(n):
        if not await one_stream(f"nat_{i}", timeout=2):
            hung.append(f"nat_{i}")
    return hung


N = 100
t0 = time.time()
hung = asyncio.run(many(N))
stop.set()
orphaned = {o.request_id for o in mgr._cb.output_router.output_queue.queue if o.is_finished()}
print(f"[part 2] {len(hung)}/{N} streams did not end within 2s (no injected delay), {time.time() - t0:.0f}s; "
      f"{len(orphaned & set(hung))} of them have their FINISHED output orphaned in output_queue "
      f"(handler registered too late); still registered handlers: {len(mgr._cb.output_router.result_handlers)}")
mgr._cb.stop(block=True, timeout=5)
