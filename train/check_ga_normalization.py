"""
Check: for a causal LM with variable-length examples (labels padded with -100), the gradient of each
optimizer step equals the gradient of (sum of token CE over the window) / (#valid shifted tokens in window),
including the final partial accumulation window.  (Spec of num_items_in_batch normalisation.)
"""

import copy
import tempfile

import torch

from transformers import DataCollatorForSeq2Seq, LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments


class Tok:  # minimal stand-in for tokenizer.pad used by DataCollatorForSeq2Seq
    padding_side = "right"
    pad_token_id = 0

    def pad(self, feats, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors="pt"):
        L = max(len(f["input_ids"]) for f in feats)
        out = {k: [] for k in feats[0]}
        for f in feats:
            n = len(f["input_ids"])
            for k, v in f.items():
                pad = 0 if k != "attention_mask" else 0
                out[k].append(list(v) + [pad] * (L - n))
        return {k: torch.tensor(v) for k, v in out.items()}


lengths = [3, 9, 5, 2, 7]
g = torch.Generator().manual_seed(0)
data = []
for i, n in enumerate(lengths):
    ids = torch.randint(1, 32, (n,), generator=g).tolist()
    data.append({"input_ids": ids, "attention_mask": [1] * n, "labels": ids, "idx": [i] * n})


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return len(data)

    def __getitem__(self, i):
        return data[i]


torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, max_position_embeddings=32)
model = LlamaForCausalLM(cfg)
ref = copy.deepcopy(model)
seen, records = [], []
orig = model.forward


def fwd(*a, idx=None, **k):
    seen.extend(sorted(set(idx[:, 0].tolist())))
    return orig(*a, **k)


model.forward = fwd


class Rec(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kw):
        records.append((list(seen), [p.grad.clone() for p in kw["model"].parameters()]))
        seen.clear()


args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=1, gradient_accumulation_steps=2,
                         num_train_epochs=1, learning_rate=0.0, max_grad_norm=0, report_to=[], use_cpu=True,
                         disable_tqdm=True, save_strategy="no", logging_strategy="no", remove_unused_columns=False)
Trainer(model=model, args=args, train_dataset=DS(), data_collator=DataCollatorForSeq2Seq(Tok()),
        callbacks=[Rec()]).train()

worst = 0.0
for win, grads in records:
    ref.zero_grad()
    tot, cnt = 0.0, 0
    for i in win:
        ids = torch.tensor([data[i]["input_ids"]])
        logits = ref(input_ids=ids).logits[0, :-1]
        tgt = ids[0, 1:]
        tot = tot + torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        cnt += tgt.numel()
    (tot / cnt).backward()
    err = max((a - p.grad).abs().max().item() for a, p in zip(grads, ref.parameters()))
    worst = max(worst, err)
    print(f"window {win}: valid tokens {cnt}, max |grad - expected| = {err:.2e}")
print("OK" if worst < 1e-5 else "MISMATCH")
