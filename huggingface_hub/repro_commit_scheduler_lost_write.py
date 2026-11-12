"""CommitScheduler loses the last append when it lands in the same mtime tick as the previous scan.

Real, unpatched huggingface_hub (main), used as documented (`with CommitScheduler(...) as scheduler:` and appends
under `with scheduler.lock:`). Stubbed: only the Hub HTTP endpoint (fakehub_commit.py).

push_to_hub decides "changed" with `last_uploaded[path] != stat.st_mtime` (_commit_scheduler.py:238) and records
the mtime seen at scan time (L244, L276). An append made after a scan but within the same timestamp tick leaves
st_mtime unchanged, so no later push (including the final one in __exit__) uploads it.

The watched folder must be on a filesystem with coarse timestamps. Here: an ext2 image with 128-byte inodes
(1 s granularity; same class as FAT/exFAT (2 s), HFS+ (1 s), ext3 with 128-byte inodes, and Linux < 6.13 local
filesystems, whose jiffy-based timestamps make appends within the same 1-4 ms tick share the mtime).
Usage: repro_commit_scheduler_lost_write.py <watched_folder_on_coarse_fs> [<watched_folder_on_fine_fs>]
"""

import os
import random
import shutil
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakehub_commit import FakeCommitHub  # noqa: E402

from huggingface_hub import CommitScheduler, HfApi  # noqa: E402


NL = b"\n"
N_NATURAL = int(os.environ.get("N_NATURAL", "10"))


def deterministic(folder):
    """A scheduled push runs, the app appends one more line right after, then the app exits."""
    shutil.rmtree(folder, ignore_errors=True)
    hub = FakeCommitHub()
    api = HfApi(endpoint=hub.endpoint, token="hf_fake")
    log = os.path.join(folder, "feedback.jsonl")
    with CommitScheduler(repo_id="user/feedback", repo_type="dataset", folder_path=folder, every=10, hf_api=api) as s:
        time.sleep(0.5)  # initial (empty) scheduled push done
        # Warm-up push (another file) so that the push below takes milliseconds, not the first-call overhead.
        with s.lock:
            with open(os.path.join(folder, "warmup.txt"), "w") as f:
                f.write("x")
        s.trigger().result()
        # Write line 1 and line 2 within the same second (the kernel's timestamp clock lags by up to a few ms,
        # so stay clear of the boundary).
        while not 0.05 < time.time() % 1 < 0.2:
            time.sleep(0.005)
        t_line1 = time.time()
        with s.lock:
            with open(log, "a") as f:
                f.write('{"line": 1}\n')
        s.trigger().result()  # the periodic push (runs every `every` minutes) happens now
        with s.lock:
            with open(log, "a") as f:
                f.write('{"line": 2}\n')
        t_line2 = time.time()
        print(f"  line 1 written at +{t_line1 % 1:.3f}s, scheduled push done, line 2 written at "
              f"+{t_line2 - t_line1:.3f}s; st_mtime after line 2 = {os.stat(log).st_mtime}, "
              f"last_uploaded = {s.last_uploaded.get(next(iter(p for p in s.last_uploaded if p.name == 'feedback.jsonl')))}")
    # __exit__ ran the final push
    local = open(log, "rb").read()
    remote = hub.remote.get("feedback.jsonl", b"")
    return local, remote, [c for c in hub.commits if c != ["warmup.txt"]]


def natural(folder, seed, seconds=4.0):
    """every=1/60 (1 s): app appends a line at random intervals (0.05-0.5 s) for 4 s, then exits the `with` block."""
    shutil.rmtree(folder, ignore_errors=True)
    hub = FakeCommitHub()
    api = HfApi(endpoint=hub.endpoint, token="hf_fake")
    log = os.path.join(folder, "feedback.jsonl")
    n = 0
    rng = random.Random(seed)
    with CommitScheduler(
        repo_id="user/feedback", repo_type="dataset", folder_path=folder, every=1 / 60, hf_api=api
    ) as s:
        t0 = time.time()
        while time.time() - t0 < seconds:
            with s.lock:
                with open(log, "a") as f:
                    f.write(f'{{"line": {n}}}\n')
            n += 1
            time.sleep(rng.uniform(0.05, 0.5))
    local = open(log, "rb").read()
    remote = hub.remote.get("feedback.jsonl", b"")
    return local, remote, hub.commits


if __name__ == "__main__":
    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    for folder in sys.argv[1:]:
        local, remote, commits = deterministic(folder)
        print(f"[deterministic] {folder}: local lines={local.count(NL)} remote lines={remote.count(NL)} "
              f"commits={len(commits)} -> {'LOST WRITE' if remote != local else 'ok'}")
        lost_runs = 0
        for i in range(N_NATURAL):
            local, remote, commits = natural(folder, seed=i)
            lost = local.count(b"\n") - remote.count(b"\n")
            lost_runs += lost > 0
            print(f"[natural run {i}] {folder}: local lines={local.count(NL)} remote lines={remote.count(NL)} "
                  f"(lost {lost}) commits={len(commits)}")
        print(f"[natural] {folder}: {lost_runs}/{N_NATURAL} runs ended with lines missing on the Hub after __exit__")
