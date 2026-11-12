"""Double resume: checkpoint after k1, resume, consume k2 more, checkpoint, resume again: the concatenation must equal
the uninterrupted run (pipelines without shuffle buffers)."""
import copy, itertools
from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from datasets.iterable_dataset import ExamplesIterable
import datasets
datasets.logging.set_verbosity_error()

def make(kind, base, n_shards, per):
    if kind == "gen":
        def g(shards):
            for s in shards:
                for j in range(per): yield f"{s}_{j}", {"a": base + s * 100 + j}
        return IterableDataset(ExamplesIterable(g, {"shards": list(range(n_shards))}))
    return Dataset.from_dict({"a": [base + s * 100 + j for s in range(n_shards) for j in range(per)]}).to_iterable_dataset(num_shards=n_shards)

async def amap(x):
    return {"a": x["a"] + 1}

def pipelines(kind):
    return {
        "inter_first": lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 2, 4)]),
        "inter_all": lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 1, 5)], stopping_strategy="all_exhausted"),
        "inter_wo": lambda: interleave_datasets([make(kind, 0, 2, 2), make(kind, 10000, 1, 5)], stopping_strategy="all_exhausted_without_replacement"),
        "inter_prob": lambda: interleave_datasets([make(kind, 0, 2, 3), make(kind, 10000, 1, 5)], probabilities=[0.4, 0.6], seed=3, stopping_strategy="all_exhausted"),
        "bmap": lambda: make(kind, 0, 3, 4).map(lambda b: {"a": [v + 1 for v in b["a"]]}, batched=True, batch_size=5),
        "amap": lambda: make(kind, 0, 3, 4).map(amap),
        "bfilter": lambda: make(kind, 0, 3, 4).filter(lambda b: [v % 3 != 1 for v in b["a"]], batched=True, batch_size=5),
        "repeat_take": lambda: make(kind, 0, 2, 3).repeat(3).take(14),
        "skip_repeat": lambda: make(kind, 0, 2, 4).skip(3).repeat(2),
        "concat_bmap": lambda: concatenate_datasets([make(kind, 0, 2, 3), make(kind, 10000, 1, 4)]).map(lambda b: {"a": [v * 2 for v in b["a"]]}, batched=True, batch_size=4),
        "batch3": lambda: make(kind, 0, 3, 4).batch(3),
    }

def val(e):
    v = e["a"]
    return tuple(v) if isinstance(v, list) else int(v)

bad = tot = 0
for kind in ["gen", "arrow"]:
    for name, mk in pipelines(kind).items():
        ref = [val(e) for e in mk()]
        n = len(ref)
        for k1 in range(1, n):
            for k2 in range(0, n - k1 + 1):
                d1 = mk(); it = iter(d1); a = [val(next(it)) for _ in range(k1)]; s1 = d1.state_dict()
                d2 = mk(); d2.load_state_dict(s1); it2 = iter(d2)
                b = []
                for _ in range(k2):
                    try: b.append(val(next(it2)))
                    except StopIteration: break
                s2 = d2.state_dict()
                d3 = mk(); d3.load_state_dict(s2); c = [val(e) for e in d3]
                tot += 1
                if a + b + c != ref:
                    bad += 1
                    if bad <= 15 or k2 > 0:
                        print(f"VIOLATION {kind} {name} k1={k1} k2={k2}: ref={ref} got={a}|{b}|{c}")
print(tot, "double resumes,", bad, "violations")
