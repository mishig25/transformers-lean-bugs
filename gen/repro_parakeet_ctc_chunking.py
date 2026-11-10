"""ASR pipeline chunking with Parakeet-CTC (no `inputs_to_logits_ratio` on the config) drops all but the first chunk.

Uses the published nvidia/parakeet-ctc-1.1b config/tokenizer/feature extractor, with the encoder shrunk to 1 small
layer and random weights (no large download). The bug is in the pipeline's stride arithmetic, not in the weights.
"""

import numpy as np
import torch

from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, ParakeetForCTC, pipeline


REPO = "nvidia/parakeet-ctc-1.1b"
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(REPO)
enc = cfg.encoder_config
enc.num_hidden_layers, enc.hidden_size, enc.intermediate_size = 1, 64, 128
enc.num_attention_heads = enc.num_key_value_heads = 2
enc.subsampling_conv_channels = 16
model = ParakeetForCTC(cfg).eval()
fe = AutoFeatureExtractor.from_pretrained(REPO)
tok = AutoTokenizer.from_pretrained(REPO)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tok, feature_extractor=fe)

print("config.inputs_to_logits_ratio:", getattr(model.config, "inputs_to_logits_ratio", "<missing>"))
print("pipe._align_to:", pipe._align_to)

rng = np.random.default_rng(0)
audio = (0.1 * rng.standard_normal(30 * 16000)).astype(np.float32)  # 30 s

# Record what postprocess receives, and replay its CTC crop (automatic_speech_recognition.py:654-661)
orig_post = pipe.postprocess
kept = []


def spy(model_outputs, **kw):
    for o in model_outputs:
        if "stride" not in o:
            continue
        n_frames = o["tokens"].shape[-1]
        total_n, left, right = o["stride"]
        kept.append((n_frames, (total_n, left, right), o["tokens"][:, left : total_n - right].shape[-1]))
    return orig_post(model_outputs, **kw)


pipe.postprocess = spy

full = pipe(audio.copy())["text"]
chunked = pipe(audio.copy(), chunk_length_s=10)["text"]
print("per-chunk (logit frames, stride passed to postprocess, frames kept after crop):")
for k in kept:
    print("   ", k)
print("frames kept in total:", sum(k[2] for k in kept), "of", model(**fe(audio, return_tensors="pt")).logits.shape[1])
print("len(text) unchunked:", len(full), " chunked (chunk_length_s=10):", len(chunked))
print("chunked text == text of first chunk's frames only:", chunked == pipe(audio[: 160000].copy())["text"])
