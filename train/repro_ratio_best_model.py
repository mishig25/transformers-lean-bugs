"""
Reproducer: ratio `eval_steps` / `save_steps` accepted by the `load_best_model_at_end` check in
TrainingArguments, but TrainerState.compute_steps rounds each ratio independently with ceil, so the
absolute save interval is not a multiple of the absolute eval interval.  The best evaluation step is then
never checkpointed and `load_best_model_at_end` loads a different (non-best) checkpoint while logging the
best score.

Setup: 25 training examples, batch size 1, 1 epoch -> max_steps = 25.
       eval_steps=0.1 (-> ceil(2.5) = 3), save_steps=0.2 (-> ceil(5.0) = 5).
compute_metrics returns a user metric; here it is scripted per evaluation so that the outcome is
deterministic: the 5th evaluation (step 15, which is also a save step) scores 0.80 and the 6th
evaluation (step 18, not a save step) scores 0.90 (the true best).
"""

import os
import tempfile

import numpy as np
import torch

from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments


class DS(torch.utils.data.Dataset):
    def __init__(self, n):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randint(0, 32, (n, 8), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"input_ids": self.x[i], "labels": self.x[i]}


scores = [0.10, 0.20, 0.30, 0.40, 0.80, 0.90, 0.50, 0.40, 0.30]  # one per evaluation
eval_calls = []


def compute_metrics(eval_pred):
    i = len(eval_calls)
    eval_calls.append(i)
    return {"score": scores[min(i, len(scores) - 1)]}


def main():
    import logging as _l

    from transformers.utils import logging as hf_logging

    hf_logging.get_logger("transformers.trainer").setLevel(_l.INFO)
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=16,
    )
    model = LlamaForCausalLM(cfg)
    out = tempfile.mkdtemp()
    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-3,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.2,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
    )
    print("TrainingArguments accepted: eval_steps =", args.eval_steps, " save_steps =", args.save_steps)

    eval_steps_seen = []

    class Rec(Trainer):
        def evaluate(self, *a, **k):
            eval_steps_seen.append(self.state.global_step)
            return super().evaluate(*a, **k)

    trainer = Rec(model=model, args=args, train_dataset=DS(25), eval_dataset=DS(4), compute_metrics=compute_metrics)
    trainer.train()
    st = trainer.state
    print("max_steps            :", st.max_steps)
    print("absolute eval_steps  :", st.eval_steps, " absolute save_steps:", st.save_steps,
          " save % eval =", st.save_steps % st.eval_steps)
    print("evaluated at steps   :", eval_steps_seen)
    hist = {h["step"]: h["eval_score"] for h in st.log_history if "eval_score" in h}
    print("eval scores          :", hist)
    best_step = max(hist, key=hist.get)
    print("best eval            : step", best_step, "score", hist[best_step])
    print("checkpoints on disk  :", sorted(int(d.split("-")[1]) for d in os.listdir(out) if d.startswith("checkpoint-")))
    print("state.best_metric    :", st.best_metric, " state.best_global_step:", st.best_global_step)
    print("state.best_model_checkpoint:", st.best_model_checkpoint)

    # Which checkpoint's weights are now in trainer.model?
    final_sd = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    from safetensors.torch import load_file

    for d in sorted(os.listdir(out)):
        if d.startswith("checkpoint-"):
            sd = load_file(os.path.join(out, d, "model.safetensors"))
            same = all(torch.equal(sd[k], final_sd[k]) for k in sd)
            if same:
                print("trainer.model after train() == weights of", d)
    ok = st.best_model_checkpoint is not None and st.best_model_checkpoint.endswith(f"checkpoint-{best_step}")
    print("EXPECTED best model loaded from checkpoint-%d -> %s" % (best_step, "OK" if ok else "BUG"))


if __name__ == "__main__":
    main()
