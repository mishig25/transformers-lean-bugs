"""Every DDP rank calls snapshot_download(repo_id, local_dir=D) then loads the model from D.

Real, unpatched huggingface_hub (main). Stubbed: only the Hub HTTP endpoint (fakehub.py: repo_info, tree listing,
HEAD/GET /resolve). Latencies: API/HEAD 0.15 s (measured real Hub HEAD: 0.11-0.29 s); the 64 MiB weights file is
streamed over 4 s. Ranks start with a random 0-100 ms skew (process start-up jitter).

Correct behaviour: each file downloaded once (the lock exists for that), and once a rank's snapshot_download has
returned, every file of the snapshot stays readable at D/<file>.
Observed on main: files are downloaded once per rank (sequentially, under the lock), and a rank that already
returned finds files missing (unlink before re-download, file_download.py:1504) while later ranks re-download.
"""

import collections
import hashlib
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakehub import FakeHub, sha256  # noqa: E402


REPO = "user/model"
COMMIT = "b" * 40
NRANKS = int(os.environ.get("NRANKS", "4"))


def rank_main(rank, endpoint, local_dir, start_at, expected, q):
    if os.environ.get("APPLY_FIX"):  # validate the proposed fix (fix_localdir.py)
        import fix_localdir

        fix_localdir.apply()
    from huggingface_hub import snapshot_download

    while time.time() < start_at:
        time.sleep(0.001)
    t0 = time.time()
    path = snapshot_download(REPO, local_dir=local_dir, endpoint=endpoint)
    t_ret = time.time()
    # "from_pretrained(local_dir)": open every file of the snapshot; repeated for 12 s (training start-up,
    # re-reads by other components, ...). Record every failure.
    failures = collections.Counter()
    first_failure = None
    while time.time() - t_ret < 12:
        for name, (size, sha) in expected.items():
            try:
                with open(os.path.join(path, name), "rb") as f:
                    n = os.fstat(f.fileno()).st_size
                if n != size:
                    failures[f"{name}: size {n} != {size}"] += 1
                    first_failure = first_failure or (round(time.time() - t_ret, 2), name, "truncated")
            except FileNotFoundError:
                failures[f"{name}: FileNotFoundError"] += 1
                first_failure = first_failure or (round(time.time() - t_ret, 2), name, "missing")
        time.sleep(0.01)
    q.put((rank, round(t_ret - t0, 2), round(t_ret - start_at, 2), dict(failures), first_failure))


def main():
    tmp = tempfile.mkdtemp(prefix="snap_ddp_")
    try:
        files = {}
        cfg = b'{"model_type": "toy"}' + b" " * 1000
        files["config.json"] = (cfg, hashlib.sha1(b"blob %d\0" % len(cfg) + cfg).hexdigest())
        tok = os.urandom(2 * 1024 * 1024)
        files["tokenizer.json"] = (tok, hashlib.sha1(b"blob %d\0" % len(tok) + tok).hexdigest())
        weights = os.urandom(64 * 1024 * 1024)
        files["model.safetensors"] = (weights, sha256(weights))
        hub = FakeHub(files, COMMIT, head_seconds=0.15)
        # Stream every GET at 16 MiB/s (64 MiB weights -> 4 s).
        hub.get_seconds_by_file = {n: len(c) / (16 * 1024 * 1024) for n, (c, _) in files.items()}
        orig_files = dict(files)

        local_dir = os.path.join(tmp, "D")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        t_start = time.time() + 4.0
        expected = {n: (len(c), sha256(c)) for n, (c, _) in orig_files.items()}
        rng = random.Random(int(os.environ.get("SEED", "0")))
        procs = [
            ctx.Process(target=rank_main, args=(r, hub.endpoint, local_dir, t_start + rng.uniform(0, 0.1), expected, q))
            for r in range(NRANKS)
        ]
        for p in procs:
            p.start()
        results = sorted((q.get(timeout=300) for _ in procs), key=lambda r: r[2])
        for p in procs:
            p.join()
        per_file = collections.Counter(name for _, name, _ in hub.gets)
        print(f"ranks={NRANKS}  GET requests per file: {dict(per_file)}  (correct: 1 each)")
        for rank, dur, ret_at, failures, first in results:
            print(f"rank {rank}: snapshot_download took {dur}s (returned at +{ret_at}s); "
                  f"failed opens after return: {sum(failures.values())} {failures if failures else ''} first={first}")
        ok = all(open(os.path.join(local_dir, n), "rb").read() == c for n, (c, _) in orig_files.items())
        print(f"final local_dir content correct: {ok}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="hfhome_")
    import atexit

    atexit.register(shutil.rmtree, os.environ["HF_HOME"], True)
    main()
