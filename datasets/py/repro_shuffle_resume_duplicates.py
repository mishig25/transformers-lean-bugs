"""Repro (finding 3): resuming a shuffled *arrow-backed* streaming dataset (parquet/json/csv/arrow, i.e. almost all
Hub datasets) replays examples that were already yielded before the checkpoint. The docs only allow examples
that sat in the shuffle buffer to be lost (docs/source/stream.mdx, IterableDataset.state_dict docstring).

Workflow: load_dataset(..., streaming=True).shuffle(seed=42, buffer_size=1000) (defaults), checkpoint with
ds.state_dict() / StatefulDataLoader.state_dict(), resume with load_state_dict().
"""
import collections, gc, os, sys, tempfile
import pyarrow as pa, pyarrow.parquet as pq
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset
import datasets
datasets.logging.set_verbosity_error()


def make_files(tmp, n_files, rows):
    files = []
    for f in range(n_files):
        p = os.path.join(tmp, f"train-{f:05d}.parquet")
        pq.write_table(pa.table({"id": list(range(f * rows, (f + 1) * rows))}), p)
        files.append(p)
    return files


def report(tag, before, after, total):
    c = collections.Counter(before + after)
    dups = sum(v - 1 for v in c.values() if v > 1)
    replayed = len(set(before) & set(after))
    missing = total - len(c)
    print(f"{tag}: seen before checkpoint={len(before)}, after resume={len(after)}, "
          f"replayed (yielded before AND after)={replayed}, total duplicates={dups}, never seen={missing}")


def plain(files, total, k=3000):
    mk = lambda: load_dataset("parquet", data_files=files, split="train", streaming=True).shuffle(seed=42, buffer_size=1000)
    ds = mk(); it = iter(ds)
    before = [next(it)["id"] for _ in range(k)]
    sd = ds.state_dict()
    ds2 = mk(); ds2.load_state_dict(sd)
    after = [ex["id"] for ex in ds2]
    report("ds.state_dict()/load_state_dict()            ", before, after, total)


def with_loader(files, total, num_workers, steps=94):
    mk = lambda: load_dataset("parquet", data_files=files, split="train", streaming=True).shuffle(seed=42, buffer_size=1000)
    dl = StatefulDataLoader(mk(), batch_size=32, num_workers=num_workers)
    it = iter(dl); before = []
    for _ in range(steps):
        before += next(it)["id"].tolist()
    sd = dl.state_dict()
    del it, dl; gc.collect()
    dl2 = StatefulDataLoader(mk(), batch_size=32, num_workers=num_workers)
    dl2.load_state_dict(sd)
    after = [x for b in dl2 for x in b["id"].tolist()]
    report(f"StatefulDataLoader(batch_size=32, num_workers={num_workers})", before, after, total)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    files = make_files(tmp, n_files=20, rows=500)  # 10_000 rows, 20 parquet shards
    plain(files, 10_000)
    with_loader(files, 10_000, num_workers=0)
    with_loader(files, 10_000, num_workers=2)   # 10 shards per worker: same tree shape, no crash (finding 1)


def sweep(files):
    mk = lambda: load_dataset("parquet", data_files=files, split="train", streaming=True).shuffle(seed=42, buffer_size=1000)
    ref_it = iter(mk())
    res = []
    for k in range(500, 9500, 250):
        ds = mk(); it = iter(ds); before = [next(it)["id"] for _ in range(k)]
        ds2 = mk(); ds2.load_state_dict(ds.state_dict())
        res.append(len(set(before) & {ex["id"] for ex in ds2}))
    print(f"checkpoint every 250 examples from 500 to 9250: replayed per resume min={min(res)} max={max(res)} "
          f"mean={sum(res)/len(res):.0f}; resumes with replays: {sum(r > 0 for r in res)}/{len(res)}")


if __name__ == "__main__" and "--sweep" in sys.argv:
    sweep(files)
