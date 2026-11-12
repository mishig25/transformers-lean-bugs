"""Repro: after resuming a StatefulDataLoader mid-epoch, every later epoch of the same IterableDataset restarts
at the checkpoint again (examples before the checkpoint are never seen again) unless set_epoch() is called.

Workflow: docs/source/stream.mdx checkpoint/resume with torchdata StatefulDataLoader, multi-epoch training loop
`for epoch in range(n): for batch in dataloader: ...` (no set_epoch: no shuffling is used).
"""
import sys
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import Dataset
import datasets
datasets.logging.set_verbosity_error()


def run(num_workers, persistent_workers):
    ds = Dataset.from_dict({"x": list(range(8))}).to_iterable_dataset(num_shards=2)
    kw = dict(batch_size=2, num_workers=num_workers)
    if num_workers:
        kw["persistent_workers"] = persistent_workers
    dl = StatefulDataLoader(ds, **kw)
    for step, batch in enumerate(dl):
        if step == 1:
            state = dl.state_dict()
            break
    # --- job restarts: new dataloader, restore checkpoint, continue training for 3 more epochs ---
    dl = StatefulDataLoader(ds, **kw)
    dl.load_state_dict(state)
    for epoch in range(3):
        xs = sorted(x for batch in dl for x in batch["x"].tolist())
        print(f"  num_workers={num_workers} persistent={persistent_workers} epoch {epoch} after resume: {xs}")


if __name__ == "__main__":
    run(0, False)
    run(2, True)
    run(2, False)
