"""Compare model-config vs processor-config fields that both sides use to count image features."""

import json

from huggingface_hub import hf_hub_download


REPOS = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llava-hf/llava-v1.6-34b-hf",
    "llava-hf/llama3-llava-next-8b-hf",
    "llava-hf/llava-next-72b-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf",
    "llava-hf/LLaVA-NeXT-Video-34B-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-hf/llava-onevision-qwen2-72b-ov-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
    "llava-hf/llava-onevision-qwen2-7b-si-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
    "ibm-granite/granite-vision-3.2-2b",
    "ibm-granite/granite-vision-3.3-2b",
    "ibm-granite/granite-4.0-3b-vision",
]


def load(repo, name):
    try:
        return json.load(open(hf_hub_download(repo, name)))
    except Exception as e:  # noqa: BLE001
        return {"__error__": type(e).__name__}


for repo in REPOS:
    cfg = load(repo, "config.json")
    pp = load(repo, "preprocessor_config.json")
    proc = load(repo, "processor_config.json")
    if "__error__" in cfg:
        print(f"{repo}: config {cfg['__error__']}")
        continue
    vc = cfg.get("vision_config", {})
    model_pin = cfg.get("image_grid_pinpoints")
    pp_pin = pp.get("image_grid_pinpoints")
    print(f"== {repo} ({cfg.get('model_type')})")
    print(
        "  pinpoints equal:", model_pin == pp_pin,
        "| model n=", None if model_pin is None else len(model_pin),
        "pp n=", None if pp_pin is None else len(pp_pin),
    )
    print(
        "  vision image_size/patch:", vc.get("image_size"), vc.get("patch_size"),
        "| pp size/crop:", pp.get("size"), pp.get("crop_size"),
    )
    print(
        "  select_strategy model/proc:", cfg.get("vision_feature_select_strategy"), proc.get("vision_feature_select_strategy"),
        "| proc patch_size:", proc.get("patch_size"),
        "| num_additional_image_tokens:", proc.get("num_additional_image_tokens"),
        "| num_image_tokens:", proc.get("num_image_tokens"),
        "| vision_aspect_ratio model/proc:", cfg.get("vision_aspect_ratio"), proc.get("vision_aspect_ratio"),
        "| downsample model/proc:", cfg.get("downsample_rate"), proc.get("downsample_rate"),
    )
    if "__error__" in proc:
        print("  processor_config:", proc["__error__"])
