"""Repro (finding 1): StatefulDataLoader(num_workers>1) checkpoint/resume crashes for streaming datasets that use
`.shuffle()` (default max_buffer_input_shards=10) or `concatenate_datasets`, with or without split_dataset_by_node.

Workflow: docs/source/stream.mdx "Save a dataset checkpoint and resume iteration" + use_with_pytorch (num_workers).
"""
import gc, json, os, sys, tempfile
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node
import datasets
datasets.logging.set_verbosity_error()


def write_files(tmp, prefix, n_files, n_rows, base):
    paths = []
    for f in range(n_files):
        p = os.path.join(tmp, f"{prefix}-{f}.jsonl")
        with open(p, "w") as fh:
            for i in range(n_rows):
                fh.write(json.dumps({"x": base + f * 100 + i}) + "\n")
        paths.append(p)
    return paths


def checkpoint_and_resume(make_ds, num_workers=2, steps=3):
    dl = StatefulDataLoader(make_ds(), batch_size=2, num_workers=num_workers)
    it = iter(dl)
    for _ in range(steps):
        next(it)
    state = dl.state_dict()
    del it, dl
    gc.collect()  # shut the first loader's workers down before the next loader forks
    dl = StatefulDataLoader(make_ds(), batch_size=2, num_workers=num_workers)
    dl.load_state_dict(state)
    try:
        n = sum(len(b["x"]) for b in dl)
        return f"OK, resumed and read {n} more examples"
    except Exception as e:
        frames = [l.strip() for l in str(e).splitlines() if "iterable_dataset.py" in l]
        last = [l.strip() for l in str(e).splitlines() if l.strip()][-1]
        return f"RESUME FAILED: {last}  ({frames[1] if len(frames) > 1 else ''})"


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    files = write_files(tmp, "train", n_files=4, n_rows=10, base=0)
    other = write_files(tmp, "extra", n_files=1, n_rows=10, base=10000)

    def stream(fs):
        return load_dataset("json", data_files=fs, split="train", streaming=True)

    cases = {
        "4 files, no shuffle": lambda: stream(files),
        "4 files, .shuffle(seed=42, buffer_size=8)": lambda: stream(files).shuffle(seed=42, buffer_size=8),
        "4 files, .shuffle(...) + split_dataset_by_node(rank=0, world_size=2)": lambda: split_dataset_by_node(
            stream(files).shuffle(seed=42, buffer_size=8), rank=0, world_size=2),
        "concatenate_datasets([4 files, 1 file])": lambda: concatenate_datasets([stream(files), stream(other)]),
        "concatenate_datasets([1 file, 1 file])": lambda: concatenate_datasets([stream(files[:1]), stream(other)]),
    }
    for name, make in cases.items():
        print(f"{name:75s} -> {checkpoint_and_resume(make)}")
        sys.stdout.flush()
