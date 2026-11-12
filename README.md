# Lean-guided bug hunt in huggingface/transformers (main @ a8d43b2, 2026-11-10)

Method: model the Python arithmetic/index logic in Lean 4 (core, no Mathlib), state the intended property as a
theorem, try to prove it; a failed proof yields a concrete counterexample, which is then reproduced in Python against
`main`. Bar (repo CLAUDE.md): only bugs that hit a published checkpoint, default config or documented workflow.

> **TLA+ (concurrency):** a second pass with TLA+ over continuous batching and `transformers serve` found 3 more
> high-priority bugs (cancel → generation-loop crash in async mode, cancel after offload → manager wedged, idle unload
> under `--continuous-batching` → model never freed / reloaded twice). Specs, reproducers and captured outputs:
> [`tla/`](tla/README.md).

> **huggingface_hub (TLA+):** concurrency findings in huggingface_hub (concurrent `local_dir` downloads deleting
> files under other ranks, a failing `upload_folder` cancelling unrelated xet transfers, …) with specs, reproducers and
> outputs: [`huggingface_hub/`](huggingface_hub/README.md).

> **datasets (TLA+):** streaming checkpoint/resume and multi-process findings in datasets (resume crash with
> `num_workers>=2`, a loaded state re-applied every epoch, shuffled resume replaying examples, …) with specs,
> reproducers and outputs: [`datasets/`](datasets/README.md).

## Setup

Reproducers target transformers `main` at commit `a8d43b20622149fab30762129c59a2b1ba829364` (2026-11-10), CPU only.

```bash
git clone https://github.com/huggingface/transformers
cd transformers && git checkout a8d43b20622149fab30762129c59a2b1ba829364 && cd ..
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ./transformers accelerate librosa pillow torchvision safetensors
```

Run a reproducer from the root of this repo, e.g. `python gen/repro_parakeet_ctc_chunking.py`. Each `gen/repro_*.out`
holds the output captured on `main`. `vlm/repro_deepseek_ocr2_no_crop.py` imports the tiny model tester from the
transformers test suite: set `TRANSFORMERS_REPO` to your checkout (default `../transformers`). Scripts download small
configs/processors from the Hub (no model weights).

Lean: install [elan](https://github.com/leanprover/elan) (Lean 4.35, core only, no Mathlib), then `lean <file>.lean`
(exit code 0 = every theorem in the file is proved).

All 16 Lean files compile with no `sorry`/`admit`/`axiom` (75 theorems). 

## Confirmed findings (reproduced on main)

| # | Severity | Area | Bug |
|---|----------|------|-----|
| 1 | High | ASR pipeline | Chunked CTC transcription (`chunk_length_s`) keeps only the first chunk for Parakeet-CTC and Granite-Speech-5-CTC |
| 2 | High | Trainer | Ratio `eval_steps`/`save_steps` + `load_best_model_at_end` silently loads a non-best checkpoint |
| 3 | Medium | Trainer | `IterableDataset` + gradient accumulation: partial window at end of a pass leaks into the next pass's first step |
| 4 | Medium | ASR pipeline | Parakeet-TDT/RNNT with `chunk_length_s`: strides dropped, overlapping audio transcribed twice |
| 5 | Medium (needs GPU confirm) | DiffusionGemma | FA path applies a ±(W-1) window in the decoder that sdpa/eager/flex and the tests do not |
| 6 | Medium-low | T5Gemma | Encoder sliding mask regressed in v5.3/5.4 from `|q-k| < W` to `|q-k| <= W` (eager = default backend) |
| 7 | Low-medium | Generation | Prompt-lookup drafts run positions past `max_length` (IndexError for absolute-position models) |
| 8 | High | Kimi-K2.5 / K2.6 | Video input never works on the native path: wrong placeholder token, wrong split sizes, short last chunk crashes |
| 9 | Medium | DeepSeek-OCR-2 | `crop_to_patches=False`: processor emits 257 placeholders, model produces 145 features → error for every image |
| 10 | Low | Idefics3 / SmolVLM | `_get_num_multimodal_tokens` (used by vLLM's Transformers backend) undercounts multi-token row/col markers |

Details for each are below (filled in from the Lean models + reproducers).

### 1. High — chunked CTC ASR drops all audio after the first chunk (Parakeet-CTC, Granite-Speech-5-CTC)
- Code: `pipelines/automatic_speech_recognition.py:374-386` (`_align_to` = `getattr(config, "inputs_to_logits_ratio", 1)`);
  strides rescaled at `:617-625`, then used to crop *logit frames* at `:654-661`. `ParakeetCTCConfig` and
  `GraniteSpeech5CTCConfig` define no `inputs_to_logits_ratio`, so strides stay in audio samples (26667) while each
  chunk has ~125 frames: chunk 1 keeps everything (incl. its right overlap), every later chunk keeps 0 frames.
- Lean: `gen/AsrChunking.lean` (sample-level tiling proved; with align=1 and 1280 samples/frame it predicts (125, 375)).
- Repro: `gen/repro_parakeet_ctc_chunking.py` → kept frames `126, 0, 0, 0, 0` (126 of 376), text 931 → 275 chars,
  chunked text == first chunk's text. `gen/repro_granite_ctc_chunking.py` → `125, 0, 0, 0, 0`, 538 → 196 chars.
- Affects: `pipeline("automatic-speech-recognition", model="nvidia/parakeet-ctc-1.1b" | "-0.6b" |
  "ibm-granite/granite-speech-5.0-470m-turboctc", chunk_length_s=...)` — the standard long-audio workflow. Silent.
- Fix: give both configs `inputs_to_logits_ratio` (1280 = subsampling × hop for these models) and handle the hop length
  in `_align_to` as already done for `lasr_ctc` (#42720); optionally raise when chunking a CTC model without the ratio.

### 2. High — ratio `eval_steps`/`save_steps` with `load_best_model_at_end` loads the wrong checkpoint
- Code: `TrainingArguments` validation (`training_args.py:1755-1768`) checks the *ratios* (0.2 is a multiple of 0.1), but
  `TrainerState.compute_steps` (`trainer_callback.py:156-167`) rounds each independently with `ceil(max_steps*ratio)`;
  `_save_checkpoint` (`trainer.py:3371-3388`) only records `best_model_checkpoint` if a checkpoint exists at the best step.
- Lean: `train/StepRatios.lean` — `spec_save_multiple_of_eval_fails` (max_steps=25: eval 3, save 5); holds only when the
  ratio divides max_steps (`save_multiple_when_exact`); proposed fix proved (`fixed_save_multiple`).
- Repro: `train/repro_ratio_best_model.py` → evals at 3,6,…,18 (best 0.9 at step 18), checkpoints at 5,10,…,25; log says
  "Loading best model from checkpoint-15 (score: 0.9)" and the final weights equal checkpoint-15.
- Affects: documented float `eval_steps`/`save_steps` with `load_best_model_at_end=True` whenever save ratio = k × eval
  ratio (k ≥ 2) and max_steps isn't divisible accordingly (most dataset sizes). Silent wrong model selection.
- Fix: derive `save_steps = eval_steps_abs * round(save_ratio / eval_ratio)` in `compute_steps`, or re-validate
  `state.save_steps % state.eval_steps == 0` after conversion.

### 3. Medium — IterableDataset + gradient accumulation leaks gradients across passes
- Code: `trainer.py:1889` `do_sync_step = (step+1) % G == 0 or (step+1) == steps_in_epoch`, with
  `steps_in_epoch = max_steps * G` for iterable datasets (`trainer.py:2546`). A pass with N % G ≠ 0 batches ends without
  `optimizer.step()`/`zero_grad()`; `step` resets to -1 and the leftover gradient joins the next pass's first window.
- Lean: `train/Accum.lean` (`windowsOK` holds for every map-style case; fails for iterable N=5, G=2).
- Repro: `train/repro_iterable_ga_leak.py` → map-style windows `[2,4] [3,0] [1]…`; iterable `[0,1] [2,3] [4,0,1]`
  with grad_norm 2.7 vs ~1.7 and logged loss 6.95 vs ~3.46 at every pass boundary.
- Affects: streaming / `IterableDataset` training with `gradient_accumulation_steps > 1` and `max_steps` > one pass.
- Fix: also sync on the last micro-batch of a window that came back short from `get_batch_samples`.

### 4. Medium — Parakeet-TDT/RNNT chunked transcription duplicates the overlaps
- Code: `tdt` branch of `_forward` (`automatic_speech_recognition.py:626-631`) pops `stride` (`:528`) and never returns it;
  postprocess concatenates every chunk's full text. No warning (unlike seq2seq).
- Repro: `gen/repro_parakeet_tdt_chunking.py` (nvidia/parakeet-tdt-0.6b-v3 config) → no chunk carries a stride; 43.3 s of
  audio transcribed for a 30 s input; output == concatenation of overlapping chunk texts.
- Affects: `chunk_length_s` with parakeet_tdt / parakeet_rnnt / Nemotron ASR (parakeet-tdt-0.6b-v3 ≈ 531k downloads).
- Fix: raise/warn for `chunk_length_s` on transducers, or carry the stride and crop by token frame positions.

### 5. Medium (CPU-emulated; needs a GPU run to confirm on the kernel) — DiffusionGemma decoder FA window
- Code: decoder attention `models/diffusion_gemma/modeling_diffusion_gemma.py:379,452` passes `sliding_window=W` with
  `is_causal=False`; `_process_flash_attention_kwargs` (`modeling_flash_attention_utils.py:687-692`) then sets
  `window_size=(W-1, W-1)` once key_length > W. Keys are `[W-1 cached ; C canvas]`, and flash-attn aligns queries
  bottom-right, so canvas query i sees only keys j ≥ i. `create_diffusion_decoder_attention_mask` and
  `test_diffusion_decoder_mask_dynamic_cache_beyond_sliding_window` require all W-1+C keys visible (sdpa/eager/flex do).
- Lean: `attn/BidirWindow.lean` — `diffusion_fa_visible_iff` (visible ⇔ j ≥ i for C ≤ W), `diffusion_fa_ne_intended`,
  `diffusion_fa_short_ok` (no effect while cached + canvas ≤ W); `#eval` on W=1024, C=256: query i misses i keys.
- Repro: `attn/repro_diffusion_gemma_fa_window.py` (tiny hub checkpoint, W=16, C=8, 40-token prompt): real
  `_process_flash_attention_kwargs` → `window_size (15, 15)`; keys dropped per canvas query `[0..7]`; decoder output
  differs by up to 1.88 vs the sdpa default when FA's documented window semantics are applied.
- Affects: google/diffusiongemma-26B-A4B-it (≈638k downloads) with `attn_implementation="flash_attention_*"` and prompts
  longer than ~W-C tokens (768 for the published config). Fix: pass `sliding_window=None` from the decoder attention.

### 6. Medium-low — T5Gemma encoder sliding window off by one (regression since v5.3/5.4)
- Code: `models/t5gemma/modeling_t5gemma.py:672` now uses `create_bidirectional_sliding_window_mask`
  (`masking_utils.py:137-147`, `abs(q-kv) <= W`). Up to v5.2.0 T5Gemma used its own
  `(q - W < kv) & (kv < q + W)`. The FA path (`sliding_window=W` → `(W-1, W-1)`) still matches the original;
  T5Gemma has `_supports_sdpa = False`, so eager (the regressed path) is the default backend.
- Lean: `attn/BidirWindow.lean` — `t5gemma_fa_eq_orig` (FA = original), `t5gemma_overlay_diff_iff` (main's mask differs
  exactly on |q-k| = W), `t5gemma_mismatch_exists` (whenever L > W), `t5gemma_short_inputs_ok` (no effect if L ≤ W).
- Repro: `attn/repro_t5gemma_window.py` → differing cells exactly the |q-k| = 4 diagonals; FA window == original; encoder
  output changes (tiny random model).
- Affects: google/t5gemma-* (e.g. 2b-2b-ul2-it ≈174k downloads, W=4096 default) only for encoder inputs > 4096 tokens;
  each query sees ≤2 extra keys. Fix: pass `sliding_window - 1` (or an exclusive overlay) for T5Gemma's mask.

### 7. Low-medium — prompt-lookup candidates exceed `max_length` in the verification forward
- Code: `generation/candidate_generator.py:1095` `end_idx = min(end_idx, input_length, self.max_length)` caps a prompt
  *index*, so the `max_length` term is dead (`gen/PromptLookup.lean`, `max_length_term_is_dead`).
- Repro: `gen/repro_prompt_lookup_positions.py` → GPT-2-style model at its 1024-position limit: greedy OK,
  `prompt_lookup_num_tokens=10` raises `IndexError`; with `max_length=204` position 210 is fed to the model.
- Earlier PRs on the over-proposal (#46993, #47195) were closed after output cropping; the crash is not covered.
  Fix: cap `end_idx` at `start_idx + (max_length - input_length - 1)`.

### 8. High — Kimi-K2.5 / K2.6 video is broken end to end (native classes)
- (a) `models/kimi_k25/processing_kimi_k25.py:46-58,92`: the fallback video token `<|kimi_k25_video_placeholder|>`
  is not in the published tokenizer (splits into 10 ids; `video_token_id` → `[UNK]`); `config.video_token_id=163840`
  is outside the vocab. Every video call raises `Mismatch in 'video' token count ... ids=[0] and text=[192]`.
- (b) `modeling_kimi_k25.py:601` `split_sizes = grid_thw.prod(-1) // 4` is t× too large for video chunks (the model pools
  over time) → `split_with_sizes ... sum exactly to 384 ... got [768, 768]`.
- (c) `get_vision_temporal_merge_index` (`modeling_kimi_k25.py:441-465`) `torch.cat`s per-chunk index blocks of
  shape (n, t, 4); a short last chunk (frames not a multiple of 4) → `Expected size 4 but got size 2`.
- Lean: `vlm/KimiK25Video.lean` (`perChunk_proc_eq_rows`, `split_overcounts`; search F=1..400: split correct only
  for F=1, cat fails for F ∈ {5,6,7,9,10,11,…}).
- Repro: `vlm/repro_kimi_k25_video.py` (published K2.6 processor + tiny model with published vision settings):
  1 frame → (a); 8 frames → (a) + (b); 6 frames → (a) + (c). Run: `python vlm/repro_kimi_k25_video.py`.
- Affects: every video input for moonshotai/Kimi-K2.5 (≈393k downloads) / K2.6 via transformers; untested in CI.
- Fix: emit `<|media_pad|>` for video patches (as Moonshot's own processor does); split by
  `grid_thw[:, 1:].prod(-1) // merge`; group/average chunks per clip before concatenating.

### 9. Medium — DeepSeek-OCR-2 `crop_to_patches=False` always fails
- `image_processing_deepseek_ocr2.py:262` resizes the global view to `tile_size` (768) when not cropping, but
  `processing_deepseek_ocr2.py:76` always counts it at `size["height"]` (1024); the model picks the 144-query path for a
  768 view (`modeling_deepseek_ocr2.py:977`). Lean: `vlm/DeepseekOcr2.lean` (equal for crop=True with any tile count;
  257 vs 145 for crop=False). Repro: `vlm/repro_deepseek_ocr2_no_crop.py` →
  `Image features and image tokens do not match, tokens: 257, features: 145` for any image.
- Fix: use `tile_size` for the global view in `replace_image_token` when `crop_to_patches` is False.

### 10. Low — Idefics3 / SmolVLM v1 `_get_num_multimodal_tokens`
- `processing_idefics3.py:272-285` assumes `<row_i_col_j>` / `<global-img>` are single tokens; they are 8–9 tokens in the
  Idefics3-8B-Llama3 / SmolVLM-Instruct tokenizers (480×640: real span 2314, helper 2227). Affects external consumers
  (vLLM Transformers backend), not the model's own check. Repro: `vlm/repro_idefics3_num_mm_tokens.py`.

### Lower priority (real, but unusual arguments or no published checkpoint)
- Gemma4 `use_bidirectional_attention="all"`: config stores `W//2 + 1` "for FA exclusive bounds", FA gets ±W/2 but the
  eager/sdpa mask (inclusive overlay) gets ±(W/2+1). No published checkpoint uses `"all"` (DiffusionGemma uses
  `"vision"`). Same FA-vs-eager pattern in `muse_glimmer_assistant` (no `+1` before FA).
- Schedulers/collators (`train/repro_scheduler_edges.py`, `train/repro_collator_edges.py`): polynomial with
  warmup == max_steps → ZeroDivisionError; `cosine_warmup_with_min_lr` with `warmup_lr_rate` and 1 warmup step never
  reaches peak LR; WSD with warmup + decay > total silently goes negative in the stable phase;
  `DataCollatorForLanguageModeling(mask_replace_prob=0.8, random_replace_prob=0.2, return_tensors="np")` →
  `ValueError: p > 1` from float64 rounding; `DataCollatorWithFlattening` with an empty example misaligns labels.

## Proved correct (coverage — no bug)
- KV cache ↔ mask contract (`attn/SlidingCache.lean`): for `DynamicSlidingWindowLayer` (with and without
  `record_past`) and `StaticSlidingWindowLayer`, every `update` returns exactly the slots
  `[kv_offset, kv_offset + kv_length)` advertised by `get_mask_sizes`, every in-window key is present, invariants are
  preserved, zero slots of static buffers lie strictly in the future (causally masked). `crop(-r)` is correct iff r ≤
  tokens appended since the last truncation — a precondition every in-tree caller (assisted decoding target/assistant,
  DeferredStopCheck) satisfies; `#eval` shows the misalignment when it is violated.
- `_ignore_causal_mask_sdpa` + sdpa `is_causal` handling (`attn/SdpaSkip.lean`): whenever the mask is skipped, SDPA's
  implicit mask equals the materialised one, for exact (dynamic) and padded (static) layouts; the
  `is_compileable and q_length == 1` guard is shown to be necessary (`guard_needed`) and is always on for StaticCache.
- Bidirectional window conventions (`attn/BidirWindow.lean`): the `sliding_window + 1` convention (ModernBERT,
  EmbeddingGemma-2, UltraBERT, NeoMME, OpenAI privacy filter) makes FA and eager/sdpa agree for every length;
  EmbeddingGemma (Gemma3 bidirectional) matches the Gemma reference and its FA path; the sdpa bidirectional skip is sound.
- Generation (`gen/`): NoRepeatNGram vectorised == dict version == spec (exhaustive, len ≤ 7, n ≤ 4); Whisper timestamp
  processor invariants (≈42k sequences); assisted-generation budgets stay within max_length.
- VLM token accounting (`vlm/Coverage.lean` + sweeps): processor count = model count for Qwen2/2.5/3-VL, GLM-4V,
  PaddleOCR-VL, MiniMax-M3 (`smart_resize` grids, sides divisible by merge size), Pixtral/Mistral3/LightOn-OCR,
  LFM2-VL, Gemma4 pooler, Phi-4-MM, LLaVA-NeXT/-Video/OneVision/Granite (`unpad_image` vs `_get_unpadded_features`),
  Nemotron-H-Omni; `_get_num_multimodal_tokens` sweeps (30 sizes, 1×1 to 4032×3024) and video frame-count sweeps
  match for the major published checkpoints; processor/model configs agree across 41 published checkpoints.
- Training (`train/`): accumulation windows tile each epoch; resume skip `(gs % U) * G` is exact (also checked with a real
  Trainer from every checkpoint); loss normalisation over accumulation incl. partial windows; linear / constant /
  polynomial(power 1) / WSD schedules hit their documented boundary values; `pad_to_multiple_of` formulas are minimal.
