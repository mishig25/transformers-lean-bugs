"""For published checkpoints, compare what the processor config implies about per-image/tile token counts
with what the model config implies (patch sizes, merge sizes, downsample ratios, tile sizes)."""

import json

from huggingface_hub import hf_hub_download


def load(repo, name):
    try:
        return json.load(open(hf_hub_download(repo, name)))
    except Exception:  # noqa: BLE001
        return {}


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def check(repo, family):
    cfg, pp, proc = load(repo, "config.json"), load(repo, "preprocessor_config.json"), load(repo, "processor_config.json")
    vpp = load(repo, "video_preprocessor_config.json")
    if isinstance(pp.get("image_processor"), dict):  # unified processor_config.json format
        pp = pp["image_processor"]
    if isinstance(proc.get("image_processor"), dict):
        pp = {**proc["image_processor"], **pp}
    if isinstance(proc.get("video_processor"), dict):
        vpp = {**proc["video_processor"], **vpp}
    vc = cfg.get("vision_config", {})
    res = {}
    if family == "idefics3":
        per_tile = (vc["image_size"] // vc["patch_size"]) ** 2 // cfg["scale_factor"] ** 2
        res = {
            "proc.image_seq_len": proc.get("image_seq_len"),
            "model tokens/tile": per_tile,
            "pp.max_image_size": pp.get("max_image_size"),
            "vision.image_size": vc["image_size"],
        }
    elif family == "internvl":
        per_tile = int((vc["image_size"][0] // vc["patch_size"][0]) ** 2 * cfg["downsample_ratio"] ** 2)
        res = {"proc.image_seq_length": proc.get("image_seq_length"), "model tokens/tile": per_tile,
               "pp.size": pp.get("size") or pp.get("crop_size"), "vision.image_size": vc["image_size"]}
    elif family == "gemma3":
        res = {"proc.image_seq_length": proc.get("image_seq_length"), "model.mm_tokens_per_image": cfg.get("mm_tokens_per_image"),
               "pp.size": pp.get("size"), "vision.image_size": vc.get("image_size")}
    elif family in ("qwen", "glm4v"):
        res = {
            "pp patch/merge/temporal": (pp.get("patch_size"), pp.get("merge_size"), pp.get("temporal_patch_size")),
            "vpp patch/merge/temporal": (vpp.get("patch_size"), vpp.get("merge_size"), vpp.get("temporal_patch_size")),
            "model patch/merge/temporal": (vc.get("patch_size"), vc.get("spatial_merge_size"), vc.get("temporal_patch_size")),
        }
    elif family == "mistral3":
        res = {"proc patch*sms": (proc.get("patch_size"), proc.get("spatial_merge_size")),
               "model patch*sms": (vc.get("patch_size"), cfg.get("spatial_merge_size")),
               "pp.patch_size": pp.get("patch_size"), "pp.size": pp.get("size")}
    elif family == "lfm2_vl":
        res = {"pp encoder_patch/downsample": (pp.get("encoder_patch_size"), pp.get("downsample_factor")),
               "model patch/downsample": (vc.get("patch_size"), cfg.get("downsample_factor"))}
    elif family == "llama4":
        res = {"proc patch/pixel_shuffle": (proc.get("patch_size"), proc.get("pixel_shuffle_ratio")),
               "model patch/pixel_shuffle": (vc.get("patch_size"), vc.get("pixel_shuffle_ratio")),
               "pp.size": pp.get("size"), "vision.image_size": vc.get("image_size")}
    elif family == "aya":
        res = {"proc img_size/patch/downsample": (proc.get("img_size"), proc.get("patch_size"), proc.get("downsample_factor")),
               "model img_size/patch/downsample": (vc.get("image_size"), vc.get("patch_size"), cfg.get("downsample_factor")),
               "pp.size": pp.get("size")}
    print(f"== {repo}")
    for k, v in res.items():
        print(f"   {k}: {v}")


CHECKS = [
    ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "idefics3"),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "idefics3"),
    ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "idefics3"),
    ("HuggingFaceTB/SmolVLM-Instruct", "idefics3"),
    ("HuggingFaceTB/SmolVLM-256M-Instruct", "idefics3"),
    ("HuggingFaceM4/Idefics3-8B-Llama3", "idefics3"),
    ("OpenGVLab/InternVL3-1B-hf", "internvl"),
    ("OpenGVLab/InternVL3-8B-hf", "internvl"),
    ("OpenGVLab/InternVL3_5-8B-HF", "internvl"),
    ("unsloth/gemma-3-4b-it", "gemma3"),
    ("Qwen/Qwen2-VL-7B-Instruct", "qwen"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen"),
    ("Qwen/Qwen3-VL-8B-Instruct", "qwen"),
    ("zai-org/GLM-4.1V-9B-Thinking", "glm4v"),
    ("zai-org/GLM-4.6V-Flash", "glm4v"),
    ("PaddlePaddle/PaddleOCR-VL", "qwen"),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "mistral3"),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral3"),
    ("LiquidAI/LFM2-VL-1.6B", "lfm2_vl"),
    ("LiquidAI/LFM2-VL-450M", "lfm2_vl"),
    ("LiquidAI/LFM2.5-VL-1.6B", "lfm2_vl"),
    ("unsloth/Llama-4-Scout-17B-16E-Instruct", "llama4"),
    ("CohereLabs/aya-vision-8b", "aya"),
]

for repo, fam in CHECKS:
    try:
        check(repo, fam)
    except Exception as e:  # noqa: BLE001
        print(f"== {repo}: error {type(e).__name__}: {e}")
