"""Resume sweep: for each pipeline and every checkpoint position k, (prefix of k examples) + (examples after
load_state_dict on a fresh copy) must equal the uninterrupted sequence. Pipelines without shuffle buffers only
(buffer loss is documented). Also runs inside simulated DataLoader workers (per-worker dataset copies, like
torch pickling the dataset into each worker)."""
import itertools, pickle, sys
import torch.utils.data as tud
from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from datasets.iterable_dataset import ExamplesIterable
from datasets.distributed import split_dataset_by_node
import datasets
datasets.logging.set_verbosity_error()

class WI:
    def __init__(self, id, num_workers): self.id, self.num_workers, self.seed, self.dataset = id, num_workers, 0, None
_orig = tud.get_worker_info
def set_worker(wid, nw):
    tud.get_worker_info = (lambda: WI(wid, nw)) if nw else (lambda: None)

def make(kind, base, n_shards, per_shard):
    ids = [base + s * 100 + j for s in range(n_shards) for j in range(per_shard)]
    if kind == "gen":
        def g(shards):
            for s in shards:
                for j in range(per_shard): yield f"{s}_{j}", {"a": base + s * 100 + j}
        return IterableDataset(ExamplesIterable(g, {"shards": list(range(n_shards))}))
    return Dataset.from_dict({"a": ids}).to_iterable_dataset(num_shards=n_shards)

async def amul(x):
    return {"a": x["a"] * 2}

def pipelines(kind):
    P = {}
    P["plain"] = lambda: make(kind, 0, 3, 4)
    P["map"] = lambda: make(kind, 0, 3, 4).map(lambda x: {"a": x["a"] + 1})
    P["bmap3"] = lambda: make(kind, 0, 3, 4).map(lambda b: {"a": [v + 1 for v in b["a"]]}, batched=True, batch_size=3)
    P["bmap_dup"] = lambda: make(kind, 0, 3, 4).map(lambda b: {"a": [v for v in b["a"] for _ in range(2)]}, batched=True, batch_size=3)
    P["amap"] = lambda: make(kind, 0, 3, 4).map(amul)
    P["filter"] = lambda: make(kind, 0, 3, 4).filter(lambda x: x["a"] % 3 != 0)
    P["bfilter"] = lambda: make(kind, 0, 3, 4).filter(lambda b: [v % 3 != 0 for v in b["a"]], batched=True, batch_size=3)
    P["skip"] = lambda: make(kind, 0, 3, 4).skip(5)
    P["take"] = lambda: make(kind, 0, 3, 4).take(7)
    P["repeat"] = lambda: make(kind, 0, 2, 3).repeat(3)
    P["inter_first"] = lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 2, 5)])
    P["inter_all"] = lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 2, 5)], stopping_strategy="all_exhausted")
    P["inter_wo"] = lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 2, 5)], stopping_strategy="all_exhausted_without_replacement")
    P["inter_prob"] = lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 2, 5)], probabilities=[0.3, 0.7], seed=1, stopping_strategy="all_exhausted")
    P["concat"] = lambda: concatenate_datasets([make(kind, 0, 2, 3), make(kind, 10000, 1, 4)])
    P["fmt_torch"] = lambda: make(kind, 0, 3, 4).with_format("torch")
    P["fmt_arrow_map"] = lambda: make(kind, 0, 3, 4).with_format("arrow").map(lambda t: t, batched=True, batch_size=3)
    P["epoch1"] = lambda: make(kind, 0, 3, 4)
    return P

def val(ex):
    v = ex["a"]
    try: return int(v)
    except TypeError: return int(v.to_pylist()[0]) if hasattr(v, "to_pylist") else int(v[0])

bad = total = 0
for kind in ["gen", "arrow"]:
    for name, mk in pipelines(kind).items():
        for world, nw in [(1, 0), (1, 2), (2, 0), (2, 2), (3, 0)]:
            for rank in range(world):
                for wid in range(max(nw, 1)):
                    def fresh():
                        ds = mk()
                        if name == "epoch1": ds.set_epoch(1)
                        if world > 1: ds = split_dataset_by_node(ds, rank=rank, world_size=world)
                        import copy; return copy.deepcopy(ds)  # like torch sending a copy of the dataset to a worker
                    set_worker(wid, nw)
                    try:
                        ref = [val(e) for e in fresh()]
                        for k in range(0, len(ref) + 1):
                            ds = fresh(); it = iter(ds); prefix = [val(next(it)) for _ in range(k)]
                            sd = ds.state_dict()
                            ds2 = fresh(); ds2.load_state_dict(sd); rest = [val(e) for e in ds2]
                            total += 1
                            if prefix + rest != ref:
                                bad += 1
                                if bad <= 40:
                                    print(f"VIOLATION {kind} {name} world={world} rank={rank} nw={nw} wid={wid} k={k}: ref={ref} prefix={prefix} rest={rest}")
                    except Exception as e:
                        print(f"ERROR {kind} {name} world={world} nw={nw}: {type(e).__name__}: {e}"); bad += 1
                    finally:
                        tud.get_worker_info = _orig
print(f"{total} checkpoints, {bad} violations")
