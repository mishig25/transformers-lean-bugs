"""
Check: resuming from checkpoint-k continues on exactly the micro-batches (and windows) the
uninterrupted run used after global step k.  L=5 examples, bs=1, G=2 (windows 2,2,1), 3 epochs.
"""

import os
import tempfile

import torch

from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments

N = 5


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 32, (N, 8), generator=g)
        return {"input_ids": x[i], "labels": x[i], "idx": torch.tensor(i)}


def run(out, resume=None):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16)
    model = LlamaForCausalLM(cfg)
    seen, windows = [], []
    orig = model.forward

    def fwd(*a, idx=None, **k):
        seen.append(int(idx.item()))
        return orig(*a, **k)

    model.forward = fwd

    class Rec(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kw):
            windows.append((state.global_step + 1, tuple(seen)))
            seen.clear()

    args = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                             num_train_epochs=3, learning_rate=1e-3, report_to=[], use_cpu=True, disable_tqdm=True,
                             save_strategy="steps", save_steps=1, logging_strategy="no", remove_unused_columns=False,
                             seed=42)
    t = Trainer(model=model, args=args, train_dataset=DS(), callbacks=[Rec()])
    t.train(resume_from_checkpoint=resume)
    return windows


out = tempfile.mkdtemp()
full = run(out)
print("uninterrupted:", full)
bad = 0
for k in range(1, len(full)):
    res = run(tempfile.mkdtemp(), resume=os.path.join(out, f"checkpoint-{k}"))
    ok = res == full[k:]
    bad += not ok
    print(f"resume from checkpoint-{k}: {'OK' if ok else 'MISMATCH ' + str(res)}")
print("ALL OK" if bad == 0 else f"{bad} MISMATCHES")
