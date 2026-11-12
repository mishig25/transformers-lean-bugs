"""`transformers serve --continuous-batching` with two models: GenerationState.get_manager() keeps ONE CB manager; a
request for model B calls stop(block=True, timeout=2) on model A's manager while A's requests are still streaming."""

import asyncio
import time

from harness import body, make_stack, make_tiny_repo

from transformers import ContinuousBatchingConfig


async def drain(resp, timeout):
    n = 0
    tail = ""

    async def go():
        nonlocal n, tail
        async for chunk in resp.body_iterator:
            n += 1
            tail = chunk[-160:]

    try:
        await asyncio.wait_for(go(), timeout)
        return f"ended after {n} chunks; last: {tail.strip()[:150]!r}"
    except asyncio.TimeoutError:
        return f"HUNG (no end after {timeout}s, {n} chunks received)"


async def main():
    repo_a, repo_b = make_tiny_repo(), make_tiny_repo()
    # small explicit cache: on CPU the default sizes the KV cache from free *system* RAM, and two managers overlap
    cfg = ContinuousBatchingConfig(num_blocks=256, block_size=32, max_batch_tokens=256)
    mm, state, handler = make_stack(model_timeout=-1, continuous_batching=True, cb_config=cfg)
    try:
        resp_a = await handler.handle_request(body(repo_a, stream=True, max_tokens=3000), "A-long")
        task_a = asyncio.create_task(drain(resp_a, timeout=60))
        await asyncio.sleep(1.0)                      # A is streaming
        t0 = time.time()
        resp_b = await handler.handle_request(body(repo_b, stream=False, max_tokens=4), "B")
        print(f"request B: {resp_b.status_code} (event loop blocked {time.time() - t0:.1f}s inside get_manager)")
        print("request A stream:", await task_a)
    finally:
        state.shutdown()
        mm.shutdown()


asyncio.run(main())
