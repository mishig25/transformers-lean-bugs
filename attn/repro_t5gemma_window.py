"""
T5Gemma encoder sliding-window regression (transformers >= 5.3/5.4, still on main).

Original T5Gemma encoder mask (<= 5.2.0, models/t5gemma/modeling_t5gemma.py):
    (q_idx - W < kv_idx) & (kv_idx < q_idx + W)          ->  |q - k| <= W - 1
main builds it with masking_utils.create_bidirectional_sliding_window_mask:
    abs(q_idx - kv_idx) <= W                              ->  |q - k| <= W
while the FA path of the same module still uses window_size=(W-1, W-1)  -> |q - k| <= W - 1.
T5Gemma has `_supports_sdpa = False`, so eager (the affected path) is the default backend.
Lean: attn/BidirWindow.lean (t5gemma_overlay_diff_iff, t5gemma_fa_eq_orig).
"""

import torch

from transformers import T5GemmaConfig, T5GemmaEncoderModel
from transformers.masking_utils import create_bidirectional_sliding_window_mask
from transformers.modeling_flash_attention_utils import _process_flash_attention_kwargs

W, L = 4, 10

cfg = T5GemmaConfig(
    encoder=dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, sliding_window=W, layer_types=["sliding_attention", "full_attention"],
    ),
    is_encoder_decoder=False,
    decoder=dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, sliding_window=W, layer_types=["sliding_attention", "full_attention"],
    ),
)
torch.manual_seed(0)
model = T5GemmaEncoderModel(cfg).eval()
print("attn implementation:", model.config._attn_implementation)
enc_cfg = model.encoder.config
inputs_embeds = torch.zeros(1, L, enc_cfg.hidden_size)
attention_mask = torch.ones(1, L, dtype=torch.long)

# 1) The mask main builds for the encoder's sliding layers (eager/sdpa/flex all use this mask function)
m = create_bidirectional_sliding_window_mask(config=enc_cfg, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
visible_main = m[0, 0] == 0 if m.dtype.is_floating_point else m[0, 0]

# 2) Original T5Gemma semantics (<= 5.2.0)
q = torch.arange(L)[:, None]
k = torch.arange(L)[None, :]
visible_orig = (q - W < k) & (k < q + W)

# 3) What the FA path of the same attention module would do
attn = model.encoder.layers[0].self_attn
fa = _process_flash_attention_kwargs(
    query_length=L, key_length=L, is_causal=attn.is_causal, sliding_window=attn.sliding_window,
    supports_mapping=__import__("collections").defaultdict(bool, {"window_size": True, "softcap": True}),
)
print("encoder layer-0 module sliding_window =", attn.sliding_window, "-> FA window_size =", fa.get("window_size"))
left, right = fa["window_size"]
visible_fa = (k >= q - left) & (k <= q + right)

diff = (visible_main != visible_orig).nonzero().tolist()
print("cells where main's eager mask != original T5Gemma mask:", diff)
print("cells where main's eager mask != main's FA window     :", (visible_main != visible_fa).nonzero().tolist())
print("FA window == original:", bool((visible_fa == visible_orig).all()))

# 4) Effect on encoder outputs: run with main's mask vs. an explicit mask with the original semantics
ids = torch.randint(1, 64, (1, L))
with torch.no_grad():
    out_main = model(input_ids=ids).last_hidden_state
    neg = torch.finfo(torch.float32).min
    orig_4d = torch.where(visible_orig, 0.0, neg)[None, None]
    full_4d = torch.zeros(1, 1, L, L)
    out_orig = model(input_ids=ids, attention_mask={"sliding_attention": orig_4d, "full_attention": full_4d}).last_hidden_state
    out_same = model(input_ids=ids, attention_mask={"sliding_attention": m, "full_attention": full_4d}).last_hidden_state
print("sanity (passing main's own mask explicitly reproduces default output):", torch.allclose(out_main, out_same))
print("max |encoder output(main) - encoder output(original window)| =", (out_main - out_orig).abs().max().item())
