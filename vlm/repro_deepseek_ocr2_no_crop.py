"""DeepSeek-OCR-2: `crop_to_patches=False` makes the processor emit 257 placeholders while the global view
is resized to `tile_size` (768) and the model yields 144 + 1 features.

Run (from this repo, transformers installed from main):  python vlm/repro_deepseek_ocr2_no_crop.py
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, DeepseekOcr2Model


sys.path.insert(0, os.environ.get("TRANSFORMERS_REPO", "../transformers"))  # transformers checkout, for its test model tester
from tests.models.deepseek_ocr2.test_modeling_deepseek_ocr2 import (  # noqa: E402
    DeepseekOcr2VisionText2TextModelTester,
)


proc = AutoProcessor.from_pretrained("deepseek-community/DeepSeek-OCR-2")
img = Image.fromarray(np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8))

for crop in (True, False):
    out = proc(images=img, text="<image>\nFree OCR.", return_tensors="pt", crop_to_patches=crop)
    n_ph = int((out["input_ids"] == proc.image_token_id).sum())
    print(
        f"crop_to_patches={crop}: pixel_values {tuple(out['pixel_values'].shape)}, "
        f"num_local_patches={out['num_local_patches'].tolist()}, placeholders={n_ph}"
    )

# Tiny random model whose SAM stem has the published 16x total stride per patch and 4x neck downsampling,
# so a 768 view -> 12x12 = 144 queries and a 1024 view -> 16x16 = 256 queries, exactly like the checkpoint.
tester = DeepseekOcr2VisionText2TextModelTester(
    parent=None, vision_sam_patch_size=16, vision_sam_window_size=4, image_token_id=proc.image_token_id,
    vocab_size=len(proc.tokenizer),
)
config = tester.get_config()
model = DeepseekOcr2Model(config).eval()

for crop in (True, False):
    out = proc(images=img, text="<image>\nFree OCR.", return_tensors="pt", crop_to_patches=crop)
    try:
        with torch.no_grad():
            model(**out)
        print(f"crop_to_patches={crop}: forward OK")
    except Exception as e:  # noqa: BLE001
        print(f"crop_to_patches={crop}: forward FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
