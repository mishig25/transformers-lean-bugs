"""
Reproducer: with an IterableDataset (no __len__) and max_steps > one pass, Trainer sets
steps_in_epoch = max_steps * gradient_accumulation_steps.  When a pass ends with a partial
accumulation window (N % G != 0), `do_sync_step = (step+1) % G == 0 or (step+1) == steps_in_epoch`
is False for the last micro-batch of the pass, so no optimizer step/zero_grad happens.  The next pass
restarts `step` at -1 and the leftover gradient is silently added to the next pass's first full window:
that optimizer step uses G + (N % G) micro-batches, each window normalised separately by its own
`num_items_in_batch`, i.e. ~2x the gradient for that step.

Setup: stream of N=5 batches per pass (batch size 1), gradient_accumulation_steps=2, max_steps=6.
Spec (same as the map-style path): each optimizer step consumes one fetched window, the partial window
at the end of a pass gets its own optimizer step (or is dropped), never merged with the next pass.
"""

import tempfile

import torch

from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments


N = 5


class Stream(torch.utils.data.IterableDataset):
    def __iter__(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 32, (N, 8), generator=g)
        for i in range(N):
            yield {"input_ids": x[i], "labels": x[i], "idx": torch.tensor(i)}


class Map(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 32, (N, 8), generator=g)
        return {"input_ids": x[i], "labels": x[i], "idx": torch.tensor(i)}


def run(ds, label):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=16,
    )
    model = LlamaForCausalLM(cfg)
    seen = []  # micro-batches whose backward ran since the last optimizer step
    windows = []

    orig_forward = model.forward

    def fwd(*a, idx=None, **k):
        seen.append(int(idx.item()))
        return orig_forward(*a, **k)

    model.forward = fwd

    class Rec(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kw):
            gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in kw["model"].parameters() if p.grad is not None))
            windows.append((state.global_step + 1, list(seen), round(gn.item(), 3)))
            seen.clear()

    args = TrainingArguments(
        output_dir=tempfile.mkdtemp(), per_device_train_batch_size=1, gradient_accumulation_steps=2,
        max_steps=6 if label == "iterable" else -1, num_train_epochs=2, learning_rate=0.0, max_grad_norm=0,
        report_to=[], use_cpu=True, disable_tqdm=True, save_strategy="no", logging_strategy="steps", logging_steps=1,
        remove_unused_columns=False,
    )
    t = Trainer(model=model, args=args, train_dataset=ds, callbacks=[Rec()])
    t.train()
    logged = {h["step"]: round(h["loss"], 3) for h in t.state.log_history if "loss" in h}
    print(f"--- {label} dataset, N={N} batches/pass, G=2 ---")
    for gs, w, gn in windows:
        print(f"  optimizer step {gs}: micro-batches {w}  grad_norm={gn}  logged loss={logged.get(gs)}")


run(Map(), "map-style")
run(Stream(), "iterable")
