"""Repro: streaming concatenate_datasets + StatefulDataLoader(num_workers=2) checkpoint/resume crashes.

Documented workflow (docs/source/stream.mdx "Save a dataset checkpoint and resume iteration"): two local JSON files
loaded with streaming=True, concatenated, fed to torchdata's StatefulDataLoader with 2 workers, checkpoint after a
few batches, resume in a new dataloader.
"""
import json, os, tempfile
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset, concatenate_datasets
import datasets
datasets.logging.set_verbosity_error()

if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    for name, base in [("a", 0), ("b", 1000)]:
        with open(os.path.join(tmp, f"{name}.jsonl"), "w") as f:
            for i in range(8):
                f.write(json.dumps({"x": base + i}) + "\n")
    a = load_dataset("json", data_files=os.path.join(tmp, "a.jsonl"), split="train", streaming=True)
    b = load_dataset("json", data_files=os.path.join(tmp, "b.jsonl"), split="train", streaming=True)
    ds = concatenate_datasets([a, b])
    print("num_shards:", ds.num_shards)

    dl = StatefulDataLoader(ds, batch_size=2, num_workers=2)
    seen = []
    for step, batch in enumerate(dl):
        seen += batch["x"].tolist()
        if step == 2:
            state = dl.state_dict()
            break
    print("before checkpoint:", seen)

    dl2 = StatefulDataLoader(ds, batch_size=2, num_workers=2)
    dl2.load_state_dict(state)
    try:
        rest = [x for batch in dl2 for x in batch["x"].tolist()]
        print("after resume:", rest)
    except Exception as e:
        print("RESUME FAILED:", type(e).__name__)
        print("\n".join(l for l in str(e).splitlines() if "iterable_dataset.py" in l or "Error" in l))
