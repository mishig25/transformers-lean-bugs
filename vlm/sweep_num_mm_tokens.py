"""Sweep: processor(text, images) placeholder count vs processor._get_num_multimodal_tokens(image_sizes).

usage: $PY sweep_num_mm_tokens.py repo [repo ...]
"""

import sys
import traceback

import numpy as np
from PIL import Image

from transformers import AutoProcessor


SIZES = [
    (224, 224), (480, 640), (640, 480), (720, 1280), (1080, 1920), (1920, 1080), (768, 1024),
    (333, 517), (517, 333), (50, 50), (28, 28), (100, 2000), (2000, 100), (3024, 4032), (4032, 3024),
    (256, 256), (512, 512), (600, 800), (1200, 1600), (375, 500), (500, 375), (1, 1), (10, 640),
    (2048, 2048), (1365, 2048), (900, 1600), (427, 640), (640, 427), (612, 612), (1536, 1024),
]


def placeholder(proc):
    for attr in ("image_token", "boi_token"):
        tok = getattr(proc, attr, None)
        if isinstance(tok, str):
            return tok
    return "<image>"


def run(repo, sizes=SIZES, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    proc = AutoProcessor.from_pretrained(repo)
    tok = placeholder(proc)
    image_token_id = getattr(proc, "image_token_id", None)
    base = f"Describe {tok} please."
    base_no_img = "Describe  please."
    n_text = len(proc.tokenizer(base_no_img, add_special_tokens=False)["input_ids"])
    mism = []
    for h, w in sizes:
        img = Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
        try:
            out = proc(text=[base], images=[[img]], return_tensors="pt", add_special_tokens=False, **extra_kwargs)
        except Exception as e:  # noqa: BLE001
            mism.append((h, w, "processor-error", f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        ids = out["input_ids"][0].numpy()
        cand = {image_token_id, proc.tokenizer.convert_tokens_to_ids(getattr(proc, "image_token", tok))}
        n_img = max(int((ids == c).sum()) for c in cand if c is not None)
        n_total = len(ids) - n_text
        try:
            mm = proc._get_num_multimodal_tokens(image_sizes=[[h, w]], **extra_kwargs)
            pred = mm["num_image_tokens"][0]
        except Exception as e:  # noqa: BLE001
            mism.append((h, w, "mm-error", f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        if pred not in (n_img, n_total):
            mism.append((h, w, f"actual image-token={n_img} total-added={n_total}", f"_get_num_multimodal_tokens={pred}"))
    print(f"== {repo}: {len(sizes) - len(mism)}/{len(sizes)} agree")
    for m in mism:
        print("   ", m)


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        try:
            run(repo)
        except Exception:  # noqa: BLE001
            print(f"== {repo}: FAILED TO RUN")
            traceback.print_exc(limit=2)
