"""Kimi-K2.5 / K2.6 video path: the model cannot consume the processor's video output.

Run (from this repo, transformers installed from main):  python vlm/repro_kimi_k25_video.py
"""

import torch

from transformers import AutoProcessor, Kimi_K25Config, Kimi_K25Model
from transformers.video_utils import VideoMetadata


torch.manual_seed(0)
REPO = "moonshotai/Kimi-K2.6"  # same processor/vision config as moonshotai/Kimi-K2.5

processor = AutoProcessor.from_pretrained(REPO)  # native Kimi_K25Processor + Kimi_K25VideoProcessor
hub_cfg = Kimi_K25Config.from_pretrained(REPO)
print("published vision config: merge_kernel_size", hub_cfg.vision_config.merge_kernel_size,
      "pos_emb_time", hub_cfg.vision_config.pos_emb_time, "| video processor temporal_patch_size",
      processor.video_processor.temporal_patch_size)

# tiny random model with the published *structural* vision settings (patch 14, merge 2x2, 4 time embeddings)
config = Kimi_K25Config(
    text_config={
        "model_type": "deepseek_v3", "vocab_size": hub_cfg.video_token_id + 8, "hidden_size": 32,
        "intermediate_size": 64, "moe_intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "kv_lora_rank": 16, "q_lora_rank": 32, "qk_rope_head_dim": 16, "v_head_dim": 32,
        "qk_nope_head_dim": 32, "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "first_k_dense_replace": 1, "n_group": 2, "topk_group": 1,
    },
    vision_config={
        "patch_size": hub_cfg.vision_config.patch_size, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 2, "pos_emb_height": 8, "pos_emb_width": 8,
        "pos_emb_time": hub_cfg.vision_config.pos_emb_time,
        "merge_kernel_size": hub_cfg.vision_config.merge_kernel_size,
    },
    projection_hidden_size=32,
    image_token_id=hub_cfg.image_token_id,
    video_token_id=hub_cfg.video_token_id,
)
model = Kimi_K25Model(config).eval()


tok = processor.tokenizer
print("processor.video_token", repr(processor.video_token), "-> tokenizes to",
      len(tok(processor.video_token, add_special_tokens=False)["input_ids"]), "ids;",
      "processor.video_token_id =", processor.video_token_id, repr(tok.convert_ids_to_tokens(processor.video_token_id)),
      "| config.video_token_id =", hub_cfg.video_token_id, "in vocab:", hub_cfg.video_token_id < len(tok))


def run(num_frames, height=336, width=448):
    video = torch.randint(0, 256, (num_frames, 3, height, width), dtype=torch.uint8)
    meta = VideoMetadata(total_num_frames=num_frames, fps=2.0, frames_indices=list(range(num_frames)))
    print(f"\n{num_frames} frames {height}x{width}:")
    # (a) full processor with the published tokenizer
    try:
        processor(text=[f"Describe {processor.video_token}"], videos=[video], video_metadata=[meta],
                  do_sample_frames=False, return_tensors="pt")
        print("  (a) processor OK")
    except Exception as e:  # noqa: BLE001
        print(f"  (a) processor FAILED: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    # (b/c) model fed the video processor output + exactly as many video placeholders as the processor intends
    vinputs = processor.video_processor(videos=[video], video_metadata=[meta], do_sample_frames=False,
                                        return_tensors="pt")
    grid = vinputs["video_grid_thw"]
    n_ph = int(sum(int(g[1] * g[2]) // 4 for g in grid))  # replace_video_token: one frame's worth per chunk
    input_ids = torch.full((1, n_ph + 4), 5)
    input_ids[0, 2 : 2 + n_ph] = config.video_token_id
    print(f"  video_grid_thw={grid.tolist()} intended placeholders={n_ph}")
    try:
        with torch.no_grad():
            model(input_ids=input_ids, pixel_values_videos=vinputs["pixel_values_videos"], video_grid_thw=grid)
        print("  (b/c) model forward OK")
    except Exception as e:  # noqa: BLE001
        print(f"  (b/c) model forward FAILED: {type(e).__name__}: {str(e).splitlines()[0][:220]}")


run(1)  # single frame (t=1): works
run(8)  # two full 4-frame chunks -> split_sizes 4x too large
run(6)  # chunks of 4 and 2 frames -> torch.cat in get_vision_temporal_merge_index fails
