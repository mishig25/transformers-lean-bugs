import copy, collections, itertools
from datasets import Dataset, IterableDataset
from datasets.iterable_dataset import ExamplesIterable
import datasets
datasets.logging.set_verbosity_error()
def make(kind, n_shards, per):
    if kind == "gen":
        def g(shards):
            for s in shards:
                for j in range(per): yield f"{s}_{j}", {"a": s * 100 + j}
        return IterableDataset(ExamplesIterable(g, {"shards": list(range(n_shards))}))
    return Dataset.from_dict({"a": [s * 100 + j for s in range(n_shards) for j in range(per)]}).to_iterable_dataset(num_shards=n_shards)
cnt = collections.Counter()
for kind, n_shards, buf, mbis, per in itertools.product(["gen", "arrow"], [1, 2, 3], [2, 3, 4, 7], [1, 10], [6, 13]):
    d = make(kind, n_shards, per).shuffle(seed=5, buffer_size=buf, max_buffer_input_shards=mbis)
    n = len(list(copy.deepcopy(d)))
    for k in range(1, n + 1):
        d1 = copy.deepcopy(d); it = iter(d1); pre = [next(it)["a"] for _ in range(k)]
        sd = d1.state_dict()
        d2 = copy.deepcopy(d); d2.load_state_dict(sd); rest = [e["a"] for e in d2]
        dups = set(pre) & set(rest)
        cnt[(kind, bool(dups))] += 1
        if dups and cnt[(kind, True)] <= 3:
            print(f"{kind} shards={n_shards} buf={buf} mbis={mbis} per={per} k={k}: prefix={pre} rest={rest} dups={sorted(dups)}")
            print("   state:", sd)
print(cnt)
