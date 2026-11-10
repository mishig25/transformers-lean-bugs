"""Prompt-lookup decoding proposes candidates past `max_length`, so the verification forward runs on positions the
model does not have. With learned absolute position embeddings (GPT-2 family, n_positions=1024) a request that fits
the context (prompt + max_new_tokens == 1024) works with plain greedy decoding but crashes with
`prompt_lookup_num_tokens`.

distilgpt2 config + tokenizer from the Hub, shrunk to 1 layer with random weights (no large download).
"""

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO = "distilbert/distilgpt2"
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(REPO)
cfg.n_layer, cfg.n_embd, cfg.n_head = 1, 64, 2
model = AutoModelForCausalLM.from_config(cfg).eval()
tok = AutoTokenizer.from_pretrained(REPO)
print("n_positions:", cfg.n_positions)

doc = (
    "The committee reviewed the quarterly report and noted that revenue grew in every region, "
    "while operating costs remained flat. The committee recommended that the board approve the budget. "
)
ids = tok(doc * 60, return_tensors="pt").input_ids[:, :1000]
max_new = cfg.n_positions - ids.shape[1]  # 24: the request fits the model's context exactly
kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)

with torch.no_grad():
    greedy = model.generate(ids, **kw)
    print("greedy: ok, output length", greedy.shape[1])
    try:
        pld = model.generate(ids, prompt_lookup_num_tokens=10, **kw)
        print("prompt lookup: ok, output length", pld.shape[1], "same as greedy:", torch.equal(pld, greedy))
    except Exception as e:
        print("prompt lookup:", type(e).__name__, str(e).splitlines()[0][:200])

# How far past max_length does the verification forward go?  Record the longest sequence the model is run on.
seen = []
h = model.transformer.register_forward_pre_hook(
    lambda m, a, k: seen.append((k.get("position_ids") if k.get("position_ids") is not None else torch.tensor([-1])).max().item()),
    with_kwargs=True,
)
small = dict(kw, max_new_tokens=4)
with torch.no_grad():
    model.generate(ids[:, :200], prompt_lookup_num_tokens=10, **small)
h.remove()
print("max_length = 204; largest position id fed to the model with prompt lookup:", max(seen))
