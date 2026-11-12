"""Repro (finding 4): Dataset.map(batched=True, drop_last_batch=True[, num_proc]) returns *unprocessed* rows for every
shard (or dataset) with fewer rows than batch_size: _map_single never initialises the writer, update_data stays None
(arrow_dataset.py:3758) and the input shard is returned unchanged (line 4070); map() then concatenates it with the
processed shards (lines 3646-3657). Expected: those rows are dropped like any other incomplete last batch."""
import tempfile, datasets
from datasets import Dataset
datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()
def f(b):
    return {"a": [x * 10 for x in b["a"]]}
if __name__ == "__main__":
    for n, bs, nproc in [(3, 4, None), (10, 4, None), (10, 4, 2), (10, 4, 3), (9, 4, 3)]:
        ds = Dataset.from_dict({"a": list(range(n))})
        try:
            out = ds.map(f, batched=True, batch_size=bs, drop_last_batch=True, num_proc=nproc)
            print(f"n={n} batch_size={bs} num_proc={nproc}: {out["a"][:]}")
        except Exception as e:
            print(f"n={n} batch_size={bs} num_proc={nproc}: {type(e).__name__}: {e}")
