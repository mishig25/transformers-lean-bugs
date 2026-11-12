"""Repro (finding 2b): IterableDataset.state_dict() right after load_state_dict() (before the resumed iterator
produced anything) returns the *start-of-epoch* state instead of the loaded checkpoint. StatefulDataLoader takes
each worker's post-resume initial state from it, so a checkpoint taken after resuming but before a worker has
produced a batch sends that worker back to the beginning of the epoch on the next resume (replayed data).

Workflow: docs/source/stream.mdx checkpoint/resume with StatefulDataLoader; a job is preempted again shortly after
resuming (or saves a checkpoint at step 0 of the resumed run).
"""
import collections, gc
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import Dataset
import datasets
datasets.logging.set_verbosity_error()


def mk():
    return Dataset.from_dict({"x": list(range(16))}).to_iterable_dataset(num_shards=4)


def run(num_workers, steps_after_resume):
    kw = dict(batch_size=2, num_workers=num_workers)
    dl = StatefulDataLoader(mk(), **kw); it = iter(dl)
    seen = [x for _ in range(4) for x in next(it)["x"].tolist()]   # 4 steps, then checkpoint #1
    sd1 = dl.state_dict(); del it, dl; gc.collect()
    dl = StatefulDataLoader(mk(), **kw); dl.load_state_dict(sd1); it = iter(dl)   # resume #1
    seen += [x for _ in range(steps_after_resume) for x in next(it)["x"].tolist()]
    sd2 = dl.state_dict(); del it, dl; gc.collect()                                # checkpoint #2
    dl = StatefulDataLoader(mk(), **kw); dl.load_state_dict(sd2)                   # resume #2
    seen += [x for b in dl for x in b["x"].tolist()]
    dups = sorted(k for k, v in collections.Counter(seen).items() if v > 1)
    missing = sorted(set(range(16)) - set(seen))
    print(f"num_workers={num_workers} steps between the two checkpoints={steps_after_resume}: "
          f"{len(seen)} examples for 16 rows, duplicated={dups}, missing={missing}")


if __name__ == "__main__":
    for nw, steps in [(0, 0), (0, 1), (2, 0), (2, 1), (2, 2)]:
        run(nw, steps)
