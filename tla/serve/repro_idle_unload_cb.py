"""`transformers serve --continuous-batching`: after an idle unload (default --model-timeout 300 s), the next request
loads a SECOND copy of the model while the CB manager keeps generating with the first one, which is never freed.
TLA+: Serve.tla, invariants OneCopy / CBUsesCached.  Here model_timeout=5 s stands in for the default 300 s."""

import asyncio
import gc
import time
import weakref

from harness import body, make_repo_once, make_stack  # noqa: F401

from transformers import Qwen2ForCausalLM


async def main():
    repo = make_repo_once()
    mm, state, handler = make_stack(model_timeout=5, continuous_batching=True)
    try:
        await scenario(repo, mm, state, handler)
    finally:
        state.shutdown()
        mm.shutdown()


async def scenario(repo, mm, state, handler):

    r1 = await handler.handle_request(body(repo, stream=False), "req-1")
    key = mm.process_model_name(repo)
    first = weakref.ref(mm.loaded_models[key].model)
    print("request 1:", r1.status_code, "| CB generates with copy #1:", state._cb_manager._cb.model is first())

    time.sleep(8)  # idle longer than model_timeout: TimedModel unloads the model
    gc.collect()
    print("after idle timeout: key still cached?", key in mm.loaded_models,
          "| copy #1 freed?", first() is None)

    r2 = await handler.handle_request(body(repo, stream=False), "req-2")
    second = mm.loaded_models[key].model
    cb_model = state._cb_manager._cb.model
    live = [o for o in gc.get_objects() if isinstance(o, Qwen2ForCausalLM)]
    print("request 2:", r2.status_code)
    print("ModelManager now serves copy #2; CB still generates with copy #1:", cb_model is first(),
          "| cb model is the cached one:", cb_model is second)
    print("live Qwen2ForCausalLM instances:", len(live))


asyncio.run(main())
