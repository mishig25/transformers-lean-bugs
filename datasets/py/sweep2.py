"""Exactly-once sweep over richer pipelines: interleave (without replacement), concatenate, map/filter,
take/skip before split, nested split_dataset_by_node, ranks x simulated DataLoader workers, epochs."""
import itertools, collections
import torch.utils.data as tud
from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from datasets.iterable_dataset import ExamplesIterable
from datasets.distributed import split_dataset_by_node
import datasets
datasets.logging.set_verbosity_error()

class WI:
    def __init__(self, id, num_workers): self.id, self.num_workers, self.seed, self.dataset = id, num_workers, 0, None
_orig = tud.get_worker_info
def run_worker(ds, wid, nw):
    tud.get_worker_info = (lambda: WI(wid, nw)) if nw else (lambda: None)
    try: return [ex["a"] for ex in ds]
    finally: tud.get_worker_info = _orig

def make(kind, base, n_shards, per_shard):
    ids = [base + s * 100 + j for s in range(n_shards) for j in range(per_shard)]
    if kind == "gen":
        def g(shards):
            for s in shards:
                for j in range(per_shard): yield f"{s}_{j}", {"a": base + s * 100 + j}
        return IterableDataset(ExamplesIterable(g, {"shards": list(range(n_shards))})), ids
    return Dataset.from_dict({"a": ids}).to_iterable_dataset(num_shards=n_shards), ids

def pipelines(kind):
    # (name, builder) -> (ds, expected multiset)
    def p_inter(sa, sb):
        def f():
            a, ea = make(kind, 0, sa, 2); b, eb = make(kind, 10000, sb, 3)
            return interleave_datasets([a, b], stopping_strategy="all_exhausted_without_replacement"), ea + eb
        return f
    def p_concat(sa, sb):
        def f():
            a, ea = make(kind, 0, sa, 2); b, eb = make(kind, 10000, sb, 3)
            return concatenate_datasets([a, b]), ea + eb
        return f
    def p_mapfilter(s):
        def f():
            a, ea = make(kind, 0, s, 4)
            ds = a.map(lambda x: {"a": x["a"] * 2}).filter(lambda x: x["a"] % 3 != 0)
            return ds, [x * 2 for x in ea if (x * 2) % 3 != 0]
        return f
    def p_batched_map(s):
        def f():
            a, ea = make(kind, 0, s, 5)
            ds = a.map(lambda b: {"a": [v + 1 for v in b["a"]]}, batched=True, batch_size=2)
            return ds, [x + 1 for x in ea]
        return f
    out = []
    for sa, sb in [(1, 2), (2, 2), (2, 3), (3, 4), (4, 4), (2, 6)]:
        out.append((f"inter{sa}x{sb}", p_inter(sa, sb)))
        out.append((f"concat{sa}+{sb}", p_concat(sa, sb)))
    for s in [1, 2, 3, 4, 6]:
        out.append((f"mapfilter{s}", p_mapfilter(s)))
        out.append((f"bmap{s}", p_batched_map(s)))
    return out

bad = total = 0
for kind in ["gen", "arrow"]:
    for name, mk in pipelines(kind):
        for world, nw, strategy, shuffle, epoch, nested in itertools.product([1, 2, 3, 4], [0, 1, 2, 3], ["auto", "shards", "examples"], [False, True], [0, 1], [False, True]):
            if nested and world not in (4,):
                continue
            base, exp = mk()
            if shuffle:
                base = base.shuffle(seed=7, buffer_size=3)
            got = []
            try:
                for rank in range(world):
                    if nested:  # 2 nodes x 2 local ranks
                        ds = split_dataset_by_node(split_dataset_by_node(base, rank // 2, 2, strategy=strategy), rank % 2, 2, strategy=strategy)
                    else:
                        ds = split_dataset_by_node(base, rank=rank, world_size=world, strategy=strategy) if world > 1 else base
                    ds.set_epoch(epoch)
                    for w in range(max(nw, 1)):
                        got += run_worker(ds, w, nw)
            except ValueError as e:
                if "strategy='shards'" in str(e) or "strategy" in str(e):
                    continue
                raise
            total += 1
            if sorted(got) != sorted(exp):
                bad += 1
                c, ce = collections.Counter(got), collections.Counter(exp)
                print(f"VIOLATION {kind} {name} world={world} nw={nw} strat={strategy} shuffle={shuffle} epoch={epoch} nested={nested}: "
                      f"extra={sorted((c - ce).elements())[:8]} missing={sorted((ce - c).elements())[:8]} (got {len(got)} exp {len(exp)})")
print(f"{total} configs, {bad} violations")
