"""Sweep: every example delivered exactly once across ranks x DataLoader workers.

Workers are simulated in-process by patching torch.utils.data.get_worker_info, which is exactly what
IterableDataset.__iter__/_iter_pytorch read (iterable_dataset.py:3057-3065, 2908-2964).
"""
import itertools, sys, collections
import torch.utils.data as tud
from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from datasets.distributed import split_dataset_by_node
import datasets
datasets.logging.set_verbosity_error()

class WI:
    def __init__(self, id, num_workers): self.id, self.num_workers, self.seed, self.dataset = id, num_workers, 0, None

_orig = tud.get_worker_info
def run_worker(ds, wid, nw):
    tud.get_worker_info = (lambda: WI(wid, nw)) if nw else (lambda: None)
    try:
        return [ex["a"] for ex in ds]
    finally:
        tud.get_worker_info = _orig

def gen(shards):
    for s in shards:
        for i in range(s * 100, s * 100 + SIZES[s]):
            yield {"a": i}

SIZES = {}
def make(kind, n_shards, per_shard):
    for s in range(n_shards): SIZES[s] = per_shard
    if kind == "gen":
        from datasets.iterable_dataset import ExamplesIterable
        return IterableDataset(ExamplesIterable(lambda shards: ((i, x) for i, x in enumerate(gen(shards))), {"shards": list(range(n_shards))}))
    else:  # arrow-backed
        ids = [s * 100 + j for s in range(n_shards) for j in range(per_shard)]
        return Dataset.from_dict({"a": ids}).to_iterable_dataset(num_shards=n_shards)

def expected(kind, n_shards, per_shard):
    return sorted(s * 100 + j for s in range(n_shards) for j in range(per_shard))

bad = 0
total = 0
for kind, n_shards, per_shard, world, nw, strategy, shuffle, epoch in itertools.product(
    ["gen", "arrow"], [1, 2, 3, 4, 6], [3], [1, 2, 3], [0, 1, 2, 3], ["auto", "shards", "examples"], [False, True], [0, 1]
):
    if strategy == "shards" and n_shards < world:
        continue
    base = make(kind, n_shards, per_shard)
    if shuffle:
        base = base.shuffle(seed=42, buffer_size=4)
    got = []
    for rank in range(world):
        ds = split_dataset_by_node(base, rank=rank, world_size=world, strategy=strategy) if world > 1 else base
        ds.set_epoch(epoch)
        for w in range(max(nw, 1)):
            got += run_worker(ds, w, nw)
    total += 1
    exp = expected(kind, n_shards, per_shard)
    if sorted(got) != exp:
        bad += 1
        c = collections.Counter(got)
        dups = sorted(k for k, v in c.items() if v > 1); missing = sorted(set(exp) - set(got))
        print(f"VIOLATION kind={kind} shards={n_shards} world={world} nw={nw} strat={strategy} shuffle={shuffle} epoch={epoch}: dups={dups} missing={missing}")
print(f"{total} configs, {bad} violations")
