"""Repro (finding 3, distributed form): DDP ranks resume a shuffled arrow-backed streaming dataset.

Workflow: load_dataset("parquet", streaming=True).shuffle(seed=42, buffer_size=1000) -> split_dataset_by_node(rank,
world_size) -> every rank consumes the same number of examples (synchronous DDP), saves its state_dict, resumes.
Ranks are simulated sequentially (split_dataset_by_node only needs rank/world_size). Counts: examples a rank
yields again after resuming (within-rank replay) and examples yielded by two ranks (cross-rank duplicates).
"""
import collections, os, sys, tempfile
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import datasets
datasets.logging.set_verbosity_error()


def run(files, world, k):
    mk = lambda r: split_dataset_by_node(
        load_dataset("parquet", data_files=files, split="train", streaming=True).shuffle(seed=42, buffer_size=1000),
        rank=r, world_size=world)
    per_rank, replayed = [], 0
    for r in range(world):
        ds = mk(r); it = iter(ds)
        before = [next(it)["id"] for _ in range(k)]
        ds2 = mk(r); ds2.load_state_dict(ds.state_dict())
        after = [ex["id"] for ex in ds2]
        replayed += len(set(before) & set(after))
        per_rank.append(set(before) | set(after))
    cnt = collections.Counter(x for s in per_rank for x in s)
    cross = sum(1 for v in cnt.values() if v > 1)
    strategy = "shards" if len(files) % world == 0 else "examples"
    print(f"{len(files)} parquet shards, world_size={world} (auto -> {strategy}), checkpoint after {k} examples/rank: "
          f"within-rank replays={replayed}, examples yielded by >1 rank={cross}, never seen={sum(1 for _ in range(len(files) * 400)) - len(cnt)}")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    files = []
    for f in range(32):
        p = os.path.join(tmp, f"train-{f:05d}.parquet")
        pq.write_table(pa.table({"id": list(range(f * 400, (f + 1) * 400))}), p)
        files.append(p)
    run(files, world=4, k=1000)        # 32 % 4 == 0 -> shards
    run(files[:30], world=4, k=1000)   # 30 % 4 != 0 -> examples (StepExamplesIterable)
