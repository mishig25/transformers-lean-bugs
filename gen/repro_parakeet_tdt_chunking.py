"""ASR pipeline chunking with Parakeet-TDT: the `tdt` branch of `_forward` pops `stride` and never returns it, so
`postprocess` concatenates the full transcription of every overlapping chunk (the stride regions are emitted twice).
Published nvidia/parakeet-tdt-0.6b-v3 config/tokenizer/feature extractor, encoder shrunk, random weights."""
import numpy as np
import torch

from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, pipeline
from transformers import GenerationConfig, ParakeetForTDT

REPO = "nvidia/parakeet-tdt-0.6b-v3"
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(REPO)
enc = cfg.encoder_config
enc.num_hidden_layers, enc.hidden_size, enc.intermediate_size = 1, 64, 128
enc.num_attention_heads = enc.num_key_value_heads = 2
enc.subsampling_conv_channels = 16
cfg.decoder_hidden_size = 64
model = ParakeetForTDT(cfg).eval()
model.generation_config = GenerationConfig.from_pretrained(REPO)
fe = AutoFeatureExtractor.from_pretrained(REPO)
tok = AutoTokenizer.from_pretrained(REPO)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tok, feature_extractor=fe)
print("pipe.type:", pipe.type)

audio = (0.1 * np.random.default_rng(0).standard_normal(30 * 16000)).astype(np.float32)

seen = []
orig_post = pipe.postprocess


def spy(model_outputs, **kw):
    seen.append([("stride" in o, o["tokens"].shape[-1]) for o in model_outputs])
    return orig_post(model_outputs, **kw)


pipe.postprocess = spy
chunked = pipe(audio.copy(), chunk_length_s=10)["text"]
print("chunks reaching postprocess: (has stride?, n_tokens) ->", seen[-1])
# Transcribe each chunk on its own, exactly as chunk_iter cuts them (chunk 160000, stride 26667, step 106666)
step, C = 160000 - 2 * 26667, 160000
starts = list(range(0, len(audio), step))
starts = starts[: next(i for i, s in enumerate(starts) if s + C >= len(audio)) + 1]
per_chunk = [pipe(audio[s : s + C].copy())["text"] for s in starts]
print("seconds of audio transcribed:", sum(min(C, len(audio) - s) for s in starts) / 16000, "for a", len(audio) / 16000, "s input")
print("chunked text == plain concatenation of the overlapping chunks' texts:", chunked == "".join(per_chunk) or chunked == " ".join(per_chunk))
