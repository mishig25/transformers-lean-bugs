"""Cache without symlinks (Windows without developer mode, or HF_HUB_DISABLE_SYMLINKS=1): a snapshot file is
returned while it is still being copied.

Real, unpatched huggingface_hub (main), HF_HUB_DISABLE_SYMLINKS=1 (same code path as Windows without symlink support).
Stubbed: only the Hub HTTP endpoint (fakehub.py).

Phase 1: two processes download file F at commit C1 at the same time. Process A downloads and *moves* the blob into
  snapshots/C1/F (_create_symlink(new_blob=True), file_download.py:721). Process B, waiting on the lock, then finds
  no blob (L1284) and downloads F a second time; the pointer exists so its blob stays in blobs/ (2 GETs).
Phase 2: the repo gets a new commit C2 in which F is unchanged (same etag). Process A' calls hf_hub_download(F, C2):
  the blob exists, so the pointer is created with shutil.copyfile(blob, snapshots/C2/F) (L1257-1260 -> L724),
  writing the final path in place. Process B' (another rank / a reader using the cache, e.g. with
  local_files_only=True or a commit-hash revision) gets snapshots/C2/F from hf_hub_download as soon as the path
  exists (L1100-1113) and reads a truncated file.
"""

import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakehub import FakeHub, sha256  # noqa: E402


REPO, F = "user/model", "model.safetensors"
C1, C2 = "1" * 40, "2" * 40
SIZE = 256 * 1024 * 1024


def download(endpoint, cache_dir, revision, start_at, q):
    from huggingface_hub import hf_hub_download

    while time.time() < start_at:
        time.sleep(0.001)
    path = hf_hub_download(REPO, F, revision=revision, cache_dir=cache_dir, endpoint=endpoint)
    q.put(("download", revision, os.path.getsize(path)))


def reader(endpoint, cache_dir, revision, start_at, q):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    while time.time() < start_at:
        time.sleep(0.001)
    while True:
        try:
            path = hf_hub_download(
                REPO, F, revision=revision, cache_dir=cache_dir, endpoint=endpoint, local_files_only=True
            )
            break
        except LocalEntryNotFoundError:
            time.sleep(0.0005)
    # Use the returned path right away (as a loader would): size + last 4 KiB (e.g. safetensors reads the header,
    # then mmaps up to the end of the file).
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        f.seek(SIZE - 4096)
        tail = f.read(4096)
    q.put(("reader", revision, size if len(tail) == 4096 else size))


def main():
    tmp = tempfile.mkdtemp(prefix="nosym_")
    try:
        cache_dir = os.path.join(tmp, "cache")
        content = os.urandom(SIZE)
        etag = sha256(content)
        hub = FakeHub({F: (content, etag)}, C1, get_seconds=1.0, head_seconds=0.15)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        # Phase 1: two concurrent downloads of C1
        t = time.time() + 3
        ps = [ctx.Process(target=download, args=(hub.endpoint, cache_dir, C1, t, q)) for _ in range(2)]
        [p.start() for p in ps]
        res = [q.get(timeout=300) for _ in ps]
        [p.join() for p in ps]
        blobs = os.path.join(cache_dir, "models--user--model", "blobs")
        print(f"phase 1: GETs={len(hub.gets)} (expected 1); results={res}; leftover blobs: {os.listdir(blobs)}")

        # Phase 2: new commit C2, F unchanged
        hub.commit = C2
        n_gets = len(hub.gets)
        t = time.time() + 3
        pa = ctx.Process(target=download, args=(hub.endpoint, cache_dir, C2, t, q))
        pb = ctx.Process(target=reader, args=(hub.endpoint, cache_dir, C2, t, q))
        pa.start()
        pb.start()
        res = [q.get(timeout=300) for _ in range(2)]
        pa.join()
        pb.join()
        print(f"phase 2: GETs={len(hub.gets) - n_gets}; results (kind, revision, bytes read)={res}; expected size {SIZE}")
        for kind, _, n in res:
            if n != SIZE:
                print(f"phase 2: {kind} got a TRUNCATED file from hf_hub_download: {n} / {SIZE} bytes")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="hfhome_")
    import atexit

    atexit.register(shutil.rmtree, os.environ["HF_HOME"], True)
    main()
