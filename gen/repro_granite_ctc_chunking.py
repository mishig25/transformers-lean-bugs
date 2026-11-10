"""Same pipeline chunking bug for GraniteSpeech5ForCTC (ibm-granite/granite-speech-5.0-470m-turboctc): config has no
`inputs_to_logits_ratio`, so `_align_to == 1` and sample-unit strides are applied to logit frames.
Published config/tokenizer/feature extractor, encoder shrunk, random weights."""
import numpy as np
import torch

from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer, pipeline

REPO = "ibm-granite/granite-speech-5.0-470m-turboctc"
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(REPO)
enc = cfg.encoder_config
enc.num_hidden_layers, enc.hidden_size, enc.intermediate_size = 2, 64, 128
enc.num_attention_heads = enc.num_key_value_heads = 2
enc.head_dim = 32
model = AutoModelForCTC.from_config(cfg).eval()
fe = AutoFeatureExtractor.from_pretrained(REPO)
tok = AutoTokenizer.from_pretrained(REPO)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tok, feature_extractor=fe)
print(type(model).__name__, "pipe.type:", pipe.type, "pipe._align_to:", pipe._align_to)

audio = (0.1 * np.random.default_rng(0).standard_normal(30 * 16000)).astype(np.float32)
orig_post, kept = pipe.postprocess, []


def spy(model_outputs, **kw):
    for o in model_outputs:
        if "stride" in o:
            total_n, left, right = o["stride"]
            kept.append((o["tokens"].shape[-1], o["stride"], o["tokens"][:, left : total_n - right].shape[-1]))
    return orig_post(model_outputs, **kw)


pipe.postprocess = spy
full = pipe(audio.copy())["text"]
chunked = pipe(audio.copy(), chunk_length_s=10)["text"]
for k in kept:
    print("   (logit frames, stride, kept)", k)
print("len(text) unchunked:", len(full), " chunked:", len(chunked))
