"""`transformers serve --continuous-batching`, streaming: CBGenerateManager.generate_streaming calls cb.add_request()
BEFORE cb.register_result_handler() (serving/utils.py). OutputRouter.deliver looks the handler up in the CB thread
when an output is produced; with no handler yet the output goes to the shared output_queue, which serve never reads.
If the finishing output is misrouted (e.g. max_tokens=1, or an immediate EOS), streamer.end() never runs and the SSE
stream never terminates. The non-streaming path registers first ("to avoid race with fast completion").
TLA+: Serve.tla, invariant NoHang (trace: Add -> CB thread delivers -> Reg -> Wait forever)."""

import asyncio
import sys
import time

from harness import body, make_repo_once, make_stack

from transformers.cli.serving import utils as serving_utils


async def consume(handler, repo, rid, timeout):
    resp = await handler.handle_request(body(repo, stream=True, max_tokens=1), rid)
    chunks = []

    async def drain():
        async for chunk in resp.body_iterator:
            chunks.append(chunk)

    try:
        await asyncio.wait_for(drain(), timeout)
        return "ok"
    except asyncio.TimeoutError:
        return "hung"


async def main(mode: str, n: int):
    repo = make_repo_once()
    mm, state, handler = make_stack(model_timeout=-1, continuous_batching=True)
    try:
        await handler.handle_request(body(repo, stream=False), "warmup")  # initialise the CB manager
        if mode == "forced":
            orig_init = serving_utils.CBStreamer.__init__

            def slow_init(self, *a, **k):
                orig_init(self, *a, **k)
                time.sleep(0.05)   # stands in for a GIL hand-off / slow tokenizer or parser init in the window

            serving_utils.CBStreamer.__init__ = slow_init
        t0 = time.time()
        results = await asyncio.gather(*(consume(handler, repo, f"{mode}-{i}", timeout=10) for i in range(n)))
        cb = state._cb_manager._cb
        print(f"[{mode}] streams: {n}, hung (no end after 10 s): {results.count('hung')}, "
              f"outputs stranded in output_queue: {cb.output_router.output_queue.qsize()}, "
              f"elapsed {time.time() - t0:.1f}s")
    finally:
        state.shutdown()
        mm.shutdown()


asyncio.run(main(sys.argv[1], int(sys.argv[2])))
