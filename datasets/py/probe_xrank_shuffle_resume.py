"""Cross-rank resume check: all ranks checkpoint after the same number k of examples (synchronous DDP), resume,
and the union over ranks of (prefix + rest) must contain no duplicate (buffer loss is allowed/documented)."""
import copy, collections, itertools
from datasets import Dataset, IterableDataset
from datasets.iterable_dataset import ExamplesIterable
from datasets.distributed import split_dataset_by_node
import datasets
datasets.logging.set_verbosity_error()

def make(kind, n_shards, per):
    if kind == "gen":
        def g(shards):
            for s in shards:
                for j in range(per): yield f"{s}_{j}", {"a": s * 100 + j}
        return IterableDataset(ExamplesIterable(g, {"shards": list(range(n_shards))}))
    return Dataset.from_dict({"a": [s * 100 + j for s in range(n_shards) for j in range(per)]}).to_iterable_dataset(num_shards=n_shards)

bad = tot = 0
for kind, n_shards, world, strat, buf, mbis, epoch in itertools.product(["gen", "arrow"], [1, 2, 3, 5], [2, 3], ["auto", "examples"], [2, 4], [1, 10], [0, 1]):
    base = make(kind, n_shards, 6).shuffle(seed=5, buffer_size=buf, max_buffer_input_shards=mbis)
    dss = [split_dataset_by_node(base, rank=r, world_size=world, strategy=strat) for r in range(world)]
    for d in dss: d.set_epoch(epoch)
    lens = [len(list(copy.deepcopy(d))) for d in dss]
    for k in range(1, min(lens) + 1):
        allv = []
        for d in dss:
            d1 = copy.deepcopy(d); it = iter(d1); pre = [next(it)["a"] for _ in range(k)]
            d2 = copy.deepcopy(d); d2.load_state_dict(d1.state_dict()); rest = [e["a"] for e in d2]
            allv += pre + rest
        tot += 1
        dups = sorted(v for v, c in collections.Counter(allv).items() if c > 1)
        if dups:
            bad += 1
            print(f"DUP {kind} shards={n_shards} world={world} strat={strat} buf={buf} mbis={mbis} epoch={epoch} k={k}: dups={dups}")
print(tot, "checks", bad, "with cross-rank duplicates")
