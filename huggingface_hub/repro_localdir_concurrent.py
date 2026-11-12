"""Concurrent hf_hub_download(..., local_dir=D) of the same file by several processes (e.g. every DDP rank, or
`hf download --local-dir` on several nodes sharing a filesystem).

Real, unpatched huggingface_hub (main). Stubbed: the Hub HTTP endpoint only (fakehub.py, serves HEAD/GET
/{repo}/resolve/{rev}/{file}; GET streamed over `get_seconds`).

Scenario "download" (file not in the HF cache):
  rank0 starts, rank1 starts 0.5 s later (both ask for the same commit). Expected: one GET, the file stays in place.
  Bug: rank1 waits for the lock, then unlinks the complete file rank0 already returned (file_download.py:1504) and
  downloads it again => 2 GETs, and the path rank0 returned is missing for the whole second download.
Scenario "copy" (file already in the HF cache, e.g. from an earlier from_pretrained):
  both ranks start together. Bug: the second rank re-copies with shutil.copyfile(cached, D/file)
  (file_download.py:1487), truncating in place the file the first rank already returned and is reading.
"""

import hashlib
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakehub import FakeHub, sha256  # noqa: E402


REPO = "user/model"
COMMIT = "a" * 40
FILE = "model.safetensors"


def worker(rank, endpoint, local_dir, cache_dir, start_at, poll_seconds, expected_sha, expected_size, q):
    # expected_sha: sha256 of the expected content
    if os.environ.get("APPLY_FIX"):  # validate the proposed fix (fix_localdir.py)
        import fix_localdir

        fix_localdir.apply()
    from huggingface_hub import hf_hub_download

    while time.time() < start_at:
        time.sleep(0.001)
    t0 = time.time()
    path = hf_hub_download(
        REPO, FILE, revision=COMMIT, local_dir=local_dir, cache_dir=cache_dir, endpoint=endpoint
    )
    t_ret = time.time()
    # The caller uses the returned path right away (e.g. from_pretrained(local_dir) / safetensors load).
    polls = missing = truncated = corrupt = 0
    missing_since = None
    max_missing_span = 0.0
    min_size = None
    while time.time() - t_ret < poll_seconds:
        polls += 1
        try:
            # Cheap check every iteration: open + size + read of the last 4 KiB (as a loader reading the last
            # tensor would). The full content is hashed once, at the end.
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                f.seek(expected_size - 4096)
                tail = f.read(4096)
                data = None
            if missing_since is not None:
                max_missing_span = max(max_missing_span, time.time() - missing_since)
                missing_since = None
            min_size = size if min_size is None else min(min_size, size)
            if size < expected_size or len(tail) < 4096 or (data is not None and len(data) < expected_size):
                truncated += 1
            elif data is not None and hashlib.sha256(data).hexdigest() != expected_sha:
                corrupt += 1
        except FileNotFoundError:
            missing += 1
            if missing_since is None:
                missing_since = time.time()
        time.sleep(0.001)
    if missing_since is not None:
        max_missing_span = max(max_missing_span, time.time() - missing_since)
    with open(path, "rb") as f:
        corrupt += hashlib.sha256(f.read()).hexdigest() != expected_sha
    q.put(
        dict(
            rank=rank,
            call_s=round(t_ret - t0, 2),
            returned_at=round(t_ret - start_at, 2),
            polls=polls,
            missing=missing,
            max_missing_s=round(max_missing_span, 2),
            truncated_reads=truncated,
            min_size_seen=min_size,
            corrupt=corrupt,
        )
    )


def run(scenario):
    tmp = tempfile.mkdtemp(prefix=f"localdir_{scenario}_")
    try:
        local_dir = os.path.join(tmp, "local_dir")
        cache_dir = os.path.join(tmp, "cache")
        if scenario == "download":
            content = os.urandom(8 * 1024 * 1024)
            get_seconds = 3.0
        else:
            content = os.urandom(int(os.environ.get("COPY_MB", "256")) * 1024 * 1024)
            get_seconds = 0.0
        if os.environ.get("ETAG_KIND", "lfs") == "lfs":
            etag = sha256(content)  # LFS/xet file: etag = sha256 of the content
        else:
            etag = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()  # regular file: git blob sha1
        hub = FakeHub({FILE: (content, etag)}, COMMIT, get_seconds=get_seconds, head_seconds=0.15)

        if scenario == "copy":
            # File already in the regular HF cache (e.g. from an earlier from_pretrained).
            from huggingface_hub import hf_hub_download

            hf_hub_download(REPO, FILE, revision=COMMIT, cache_dir=cache_dir, endpoint=hub.endpoint)
            print(f"[copy] pre-populated HF cache ({len(hub.gets)} GET)")
            hub.gets.clear()

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        t_start = time.time() + 3.0  # leave time for the interpreters to start
        offsets = [float(x) for x in os.environ.get("OFFSETS", "0,0").split(",")]
        procs = [
            ctx.Process(
                target=worker,
                args=(r, hub.endpoint, local_dir, cache_dir, t_start + off, 6.0, sha256(content), len(content), q),
            )
            for r, off in enumerate(offsets)
        ]
        for p in procs:
            p.start()
        results = sorted((q.get(timeout=120) for _ in procs), key=lambda r: r["returned_at"])
        for p in procs:
            p.join()
        print(f"[{scenario}] GET requests for {FILE}: {len(hub.gets)} (correct: {1 if scenario == 'download' else 0})")
        for r in results:
            print(f"[{scenario}] {r}")
        final = open(os.path.join(local_dir, FILE), "rb").read()
        print(f"[{scenario}] final file correct: {final == content}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="hfhome_")
    import atexit

    atexit.register(shutil.rmtree, os.environ["HF_HOME"], True)
    for s in sys.argv[1:] or ["download", "copy"]:
        run(s)
