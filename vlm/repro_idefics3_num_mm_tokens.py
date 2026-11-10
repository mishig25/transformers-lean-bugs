"""Idefics3-8B-Llama3: `_get_num_multimodal_tokens` (used by vLLM's Transformers backend) assumes each
`<row_i_col_j>` / `<global-img>` marker is one token, but this checkpoint's tokenizer splits them.
The `<image>` count itself (what the model checks) is correct.

Run (from this repo, transformers installed from main):  python vlm/repro_idefics3_num_mm_tokens.py
"""

import numpy as np
from PIL import Image

from transformers import AutoProcessor


for repo in ("HuggingFaceM4/Idefics3-8B-Llama3", "HuggingFaceTB/SmolVLM-Instruct"):
    p = AutoProcessor.from_pretrained(repo)
    print(f"== {repo}: '<row_1_col_1>' -> {len(p.tokenizer('<row_1_col_1>', add_special_tokens=False)['input_ids'])} ids")
    if not hasattr(p, "_get_num_multimodal_tokens"):
        print("   (no _get_num_multimodal_tokens)")
        continue
    for h, w in [(480, 640), (224, 224), (1080, 1920)]:
        img = Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
        ids = p(text=["<image>"], images=[[img]], add_special_tokens=False)["input_ids"][0]
        pred = p._get_num_multimodal_tokens(image_sizes=[[h, w]])["num_image_tokens"][0]
        print(f"   {h}x{w}: placeholder span={len(ids)} (of which <image>={ids.count(p.image_token_id)}), "
              f"_get_num_multimodal_tokens={pred}")
