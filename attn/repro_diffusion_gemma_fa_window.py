"""
DiffusionGemma decoder: the flash-attention path applies a sliding window that the other backends (and the tests)
do not apply.

The decoder's sliding layers attend from the C canvas tokens to [W-1 cached encoder keys ; C canvas keys].
`create_diffusion_decoder_attention_mask` and tests/models/diffusion_gemma (`expected_non_zero_sliding`) make every
canvas query see all W-1+C keys (sdpa / eager / flex). But the attention module passes `sliding_window=W` with
`is_causal=False`, so `_process_flash_attention_kwargs` sets `window_size=(W-1, W-1)` as soon as key_length > W, and
flash-attn aligns queries bottom-right: canvas query i then only sees keys j >= i, dropping the i oldest cached keys.
Lean: attn/BidirWindow.lean (diffusion_fa_visible_iff, diffusion_fa_ne_intended).

No GPU here: we (1) call the real `_process_flash_attention_kwargs` with the decoder layer's arguments, and
(2) emulate flash-attn's documented window semantics ("query i attends keys in
[i + seqlen_k - seqlen_q - left, i + seqlen_k - seqlen_q + right]") with an explicit mask under sdpa.
"""

import collections

import torch

from transformers import AutoConfig, DiffusionGemmaForBlockDiffusion, DynamicCache
from transformers.modeling_flash_attention_utils import _process_flash_attention_kwargs


REPO = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"
W, C, PROMPT = 16, 8, 40  # shrink the window so a short prompt overflows it (published: W=1024, C=256)

config = AutoConfig.from_pretrained(REPO)
config.text_config.sliding_window = W
config.canvas_length = C
torch.manual_seed(0)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(REPO, config=config, attn_implementation="sdpa").eval()
dec = model.model.decoder

prompt = torch.randint(10, 200, (1, PROMPT))
canvas = torch.randint(10, 200, (1, C))
with torch.no_grad():
    cache = model.model.encoder(input_ids=prompt, past_key_values=DynamicCache(config=config)).past_key_values

sl = next(i for i, t in enumerate(config.text_config.layer_types) if t == "sliding_attention")
attn = dec.layers[sl].self_attn
klen = cache.layers[sl].keys.shape[-2] + C
fa = _process_flash_attention_kwargs(
    query_length=C, key_length=klen, is_causal=attn.is_causal, sliding_window=attn.sliding_window,
    supports_mapping=collections.defaultdict(bool, window_size=True),
)
print(f"sliding layer: cached keys={klen - C}, canvas={C}, key_length={klen}, module sliding_window={attn.sliding_window}")
print("FA kwargs window_size:", fa.get("window_size"), " causal:", fa["causal"])

left, right = fa["window_size"]
i = torch.arange(C)[:, None]
j = torch.arange(klen)[None, :]
fa_visible = (j >= i + klen - C - left) & (j <= i + klen - C + right)
print("keys dropped per canvas query under FA:", (~fa_visible).sum(-1).tolist())

with torch.no_grad():
    default = dec(decoder_input_ids=canvas, past_key_values=cache).last_hidden_state
    full_kv = cache.layers[[t == "full_attention" for t in config.text_config.layer_types].index(True)].keys.shape[-2] + C
    masks = {
        "sliding_attention": fa_visible[None, None],
        "full_attention": torch.ones(1, 1, C, full_kv, dtype=torch.bool),
    }
    fa_emulated = dec(decoder_input_ids=canvas, past_key_values=cache, decoder_attention_mask=masks).last_hidden_state
    all_visible = {"sliding_attention": torch.ones(1, 1, C, klen, dtype=torch.bool), "full_attention": masks["full_attention"]}
    same = dec(decoder_input_ids=canvas, past_key_values=cache, decoder_attention_mask=all_visible).last_hidden_state
print("sanity: explicit all-visible mask reproduces the sdpa default:", torch.allclose(default, same, atol=1e-6))
print("max |decoder output (sdpa default) - decoder output (FA window semantics)| =", (default - fa_emulated).abs().max().item())
