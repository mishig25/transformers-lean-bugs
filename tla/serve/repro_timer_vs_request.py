"""`transformers serve`: TimedModel.delete_model (idle-timeout Timer thread) runs without the per-model lock and sets
`self.model = None` BEFORE gc.collect()/empty_cache() and only then pops the cache entry (on_unload). A request
arriving in that window takes the cached branch of load_model_and_processor and gets model=None.
TLA+: Serve.tla, invariant NoFail (trace: L1 r1 -> T0 -> T1 -> ... -> L1/L2 r2 reads None).
The window is stretched here (slow gc.collect, as on a large heap) to make the interleaving deterministic."""

import asyncio
import threading
import time

from harness import body, make_repo_once, make_stack

from transformers.cli.serving import model_manager as mm_mod


async def main():
    repo = make_repo_once()
    mm, state, handler = make_stack(model_timeout=3, continuous_batching=False)
    in_window = threading.Event()
    real_collect = mm_mod.gc.collect

    def slow_collect(*a, **k):
        in_window.set()
        time.sleep(1.0)           # a full collection on a big heap + empty_cache can take this long
        return real_collect(*a, **k)

    mm_mod.gc.collect = slow_collect
    try:
        r1 = await handler.handle_request(body(repo, stream=False), "req-1")
        print("request 1:", r1.status_code)
        await asyncio.to_thread(in_window.wait, 10)   # idle until the timeout fires and delete_model starts
        key = mm.process_model_name(repo)
        print("timer fired: key still in loaded_models:", key in mm.loaded_models,
              "| its .model:", mm.loaded_models[key].model if key in mm.loaded_models else "-")
        try:
            r2 = await handler.handle_request(body(repo, stream=False), "req-2")
            print("request 2:", r2.status_code)
        except Exception as e:
            print("request 2 raised:", type(e).__name__, str(e)[:120])
    finally:
        mm_mod.gc.collect = real_collect
        state.shutdown()
        mm.shutdown()


asyncio.run(main())
